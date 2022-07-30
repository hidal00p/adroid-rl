"""
Defines a heuristic for features that could be extracted from a trained convolutional network.
"""

import pybullet as p
from aviary.CustomAviary import CustomAviary
from utils import TrigConsts

class VisionParams:
    """
    Configurable helper data strucutre that definies ObstacleSensor.class sensory params.
    Accuracy decay rate, vision cone range (in degrees), number of sensitive measurment segments.
    """
    def __init__(self, visionAngle = 160, accuracyDecayRate = 0.8, nSegments = 8):
        self.visionAngle = visionAngle * TrigConsts.DEG2RAD
        self.accuracyDecayRate = accuracyDecayRate
        self.nSegments = nSegments

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
    def __init__(self, env : CustomAviary, visionParams = VisionParams()):
        
        assert isinstance(env, CustomAviary)
        
        self.env = env
        self.visionParams = visionParams
    
    def _detectObstacles(self):
        posA = self.env._extractAgentPosition()
        orVecA = self.env._extractAgentOrientationVector()
        obPosVec = self.env.PILLAR_DATA

        # TODO: construct a sensor that operates within this plane in range [-alpha, alpha],
        # where alpha = visionParams.visionAngle / 2, and is counted from orVecA as a zero radian axis
        norm, shift = self.env._extractAgentOrientationPlane()
        print(norm, shift)

    """
    Returns an (1, self.visionParams.nSegment) vector of observations, that identify proximity to objects.
    """
    def computeMeasurement(self):
        pass
        