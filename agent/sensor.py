"""
Defines a heuristic for features that could be extracted from a trained convolutional network.
"""

import gym
from gym import spaces
import pybullet as p
import numpy as np

from utils import TrigConsts

class VisionParams:
    """
    Configurable helper data strucutre that definies ObstacleSensor.class sensory params.
    Accuracy decay rate, vision cone range (in degrees), number of sensitive measurment segments.
    """

    """
    * visionAngle (degs) - breadth of vision field of the sensor
    * nSegments (uint) - number of discrete rays that will be emmitted to gather measurments in the vision field

    (Note that nSegments parameter has to be complient with neural network input layer)
    """
    def __init__(self, visionAngle = 160, nSegments = 20, range = .65):
        self.visionAngle = visionAngle * TrigConsts.DEG2RAD
        self.nSegments = nSegments
        self.range = range

class ObstacleSensor():
    """
    Gets access to obstacle and agent ids.
    Uses obstacle and agent position to measure distance. Uses agent orientation to measure the direction of sending sensory signal.
    With this knowledge a sensory measurement may be computed, which gives a vector of length N for which a segments intensity is proportional to the distance to the obstacle.
    Measurment accuracy should drop with distance.
    """
    
    """
    * env - a configured CustomAviary with obstacle forest and agent
    * visionParams - VisionParams.class instance
    """
    def __init__(self, env : gym.Env, visionParams = VisionParams()):
        self.env = env
        self.visionParams = visionParams

    """
    Computes a pencil of rays that get emmited to obtain data about obstacles.
    Observation parameters are constructed with respect to agent's orientation, vision angle and number of vision segments.

    Output: list of end coordiantes [[c_1], [c_2], ..., [c_nSegments]]
    """
    def _calculateRayPencil(self):
        # [x. y] themselves form a vector space equivalent to [[1, 0].T, [0, 1].T]
        # However, matrix formed by x, y stacked is a transformation matrix from local vector space to world R^3 coordinates
        rayPencil = []
        x, y, posA = self.env._extractAgentOrientationPlaneParams()
        
        # 2x3
        transformMatrix = np.array(
            [
                x, 
                y
            ]
        )
        
        step = self.visionParams.visionAngle / self.visionParams.nSegments
        currentAngle = - self.visionParams.visionAngle / 2
        for _ in range(self.visionParams.nSegments):
            # perform computation at 
            computationAngle = currentAngle + (step / 2)

            # 1x2
            visionRayMatrixLocal = np.array(
                [
                    np.cos(computationAngle), np.sin(computationAngle)
                ]
            )

            # 1x3 visionRayMatrixLocal represented in the world frame coordinates
            visionRayMatrixWorld = np.matmul(visionRayMatrixLocal, transformMatrix)
            
            # appends a tuple of visionRayMatrixLocal scaled to sensor range
            # and angle of detection
            rayPencil.append(
                ( (self.visionParams.range * visionRayMatrixWorld) + posA, currentAngle )
            )
            
            currentAngle = currentAngle + step
        
        return rayPencil
    
    def _computeDitance(self, x, y):
        x = np.array(x)
        y = np.array(y)

    """
    Takes advantage of raytestBatch API exposed by pybullet physics engine
    """
    def _detectObstacles(self):
        posA = self.env._extractAgentPosition()
        rayPencil = self._calculateRayPencil()
        
        measurments = p.rayTestBatch(
            rayFromPositions = [posA for _ in rayPencil],
            rayToPositions = [rayFinalCoord for rayFinalCoord, _ in rayPencil]
        )

        i = 0
        postprocessedMeasurments = []
        for _, _, _, hitPos, _ in measurments:
            # Compute a hit distance in agent's reference frame
            hitDistance =  np.linalg.norm(np.array(list(hitPos)) - np.array(posA))
            _, hitAngle = rayPencil[i]
            postprocessedMeasurments.append([hitDistance, hitAngle])
            i += 1
            
        return postprocessedMeasurments
    
    """
    Flattened vector ready for usage in NN input layer
    """
    def getReadyReadings(self):
        return np.array(self._detectObstacles()).flatten()

    def observationSpace(self):
        minAngle = -self.visionParams.visionAngle / 2
        maxAngle = -1 * minAngle
        minDistance = 0.0
        maxDistance = self.visionParams.range

        minObservationVector = [[minDistance, minAngle] for _ in range(self.visionParams.nSegments)]
        maxObservationVector = [[maxDistance, maxAngle] for _ in range(self.visionParams.nSegments)]

        return spaces.Box(
            low=np.array(minObservationVector).flatten(),
            high=np.array(maxObservationVector).flatten(),
            dtype=np.float32
        )