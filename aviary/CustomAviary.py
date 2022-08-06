from enum import Enum
import numpy as np
from gym import spaces
import pkg_resources
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import BaseSingleAgentAviary, ActionType
from agent.sensor import ObstacleSensor, VisionParams
from aviary.train import TrainingConfig

from utils import ForestProvider, OrientationVec, RewardBuffer

class CustomAviary(BaseSingleAgentAviary):
    ABSOLUTE_PENALTY = -30
    BAIT_SPHERE_RADIUS = .2
    BOUNDARY_RADIUS = 2.4
    """
    Custom aviary inherits from BaseSingleAgentAviary
    """

    def _weighted(weight):
        
        def __weighted(func):

            def weighted(self):
                return weight * func(self)
            
            return weighted
        
        return __weighted

    # Decorators
    def _logged(obj):

        def __logged(func):

            def logged(self):
                print(f"[LOG INFO] {obj}:")
                func(self)
        
            return logged

        return __logged

    """
    Currently CustomAviary only works on the basis of OnservationType.KIN. It combines this observation vector
    """ 
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 forestProvider: ForestProvider = None,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False,
                 act: ActionType=ActionType.VEL,
                 visionParams: VisionParams=VisionParams(),
                 criticalDistance: float = .075, # if critical distance is too small float32 may overflow
                 fTimeComponent: bool = False,
                 fDebug: bool = False,
                 fStrictDeath: bool = False,
                 baitResetFrequency: int = 10,
                 avEpisodeSteps: int = 2400, # equivalent to 10 sec
                ):
                 """
                 CustomAviary class operates an agent, which attempts to navigate through a forest of obstacles.
                 """
                 
                 assert forestProvider != None, "Please provide a correctly configured forest"

                 self.PILLAR_DATA : list((float, float, int)) = [] # tuple of format (x_coord, y_coord, pybullet_id)
                 self.forestProvider = forestProvider
                 self.obstacleSensor = ObstacleSensor(self, visionParams=visionParams)
                 self.finito = False
                 self.fDebug = fDebug
                 self.fTimeComponent = fTimeComponent
                 
                 self.criticalDistance = criticalDistance
                 self.BOUNDARY_RADIUS = np.sqrt(2)*(self.forestProvider.forestSize + self.forestProvider.x_offset)
                 
                 self.fStrictDeath = fStrictDeath

                 self.baitResetFrequency = baitResetFrequency
                 self.baitResetCounter = 1

                 if (self.fDebug):
                    self.rewardBuffer = RewardBuffer()

                 super().__init__(
                    drone_model=drone_model,
                    initial_xyzs=initial_xyzs,
                    initial_rpys=initial_rpys,
                    physics=physics,
                    freq=freq,
                    aggregate_phy_steps=aggregate_phy_steps,
                    gui=gui,
                    record=record,
                    act=act
                 )
                 self.EPISODE_LEN_SEC = avEpisodeSteps / freq
    
    def _resetPositionToRand(self):
        xBoundary, yBoundary, _ = self.forestProvider.getPoissonForrestGeometry()
        xMin, xMax = xBoundary
        yMin, yMax = yBoundary
        zInit = self.INIT_XYZS[0][2]
        self.INIT_XYZS = np.array([[np.random.uniform(xMin, xMax), np.random.uniform(yMin, yMax), zInit]])

    """
    Section related to spawning obstacles
    """
    def _resetForest(self):
        self.forestProvider._generatePoissonForest()
        if self.baitResetCounter %  self.baitResetFrequency == 0:
            self.forestProvider.resetBaitPosition()
        self.baitResetCounter += 1

    def _generatePillar(self, coords):
        x, y = coords
        id = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/column.urdf'),
            [x, y, .4 / 2],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT
        )
        
        self.PILLAR_DATA.append((x, y, id))
    
    def _addBait(self):
        x, y = self.forestProvider.baitCoordinates
        self.BAIT_ID = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/bait.urdf'),
            [x, y, .025 / 2],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT
        )

    def _addObstacles(self):
        # Add bait
        self._addBait()
        
        # NOTE
        # Add forest
        # for root in self.forestProvider.forestGrid:
        #     self._generatePillar(tuple(root))
    
    """
    Section related to querying simulator for position and orientation of simulated objects
    """
    def _getPosAndOrient(bodyId, clientId):
        pos, quat = p.getBasePositionAndOrientation(bodyId, clientId)
        rpy, orientationMatrix = p.getEulerFromQuaternion(quat), p.getMatrixFromQuaternion(quat)
        
        x = orientationMatrix[0:3]
        y = orientationMatrix[3:6]
        z = orientationMatrix[6:9]
        
        orientationMatrix = np.array([x, y, z]).T
        return (pos, rpy, orientationMatrix)
    
    def _extractAgentOrientationPlaneParams(self):
        posA, _, orientationMatrix = CustomAviary._getPosAndOrient(self.DRONE_IDS[0], self.CLIENT)
        
        # Returns norm that describes the plane n (a, b, c), and offset point P (x0, y0, z0)
        # which forms a plane a(x - x0) + b(y - y0) + c(z - z0) = 0,
        # or ax + by + cz = d, where d = ax0 + by0 + cz0
        return (orientationMatrix.T[0], orientationMatrix.T[1], posA)

    def _extractBaitPosition(self):
        pos, _, _ = CustomAviary._getPosAndOrient(self.BAIT_ID, self.CLIENT)
        return np.array(pos)
    
    def _extractAgentPosition(self):
        pos, _, _ = CustomAviary._getPosAndOrient(self.DRONE_IDS[0], self.CLIENT)
        return np.array(pos)

    def _extractAgentOrientationVector(self) -> OrientationVec:
        _, _, orientationMatrix = CustomAviary._getPosAndOrient(self.DRONE_IDS[0], self.CLIENT)
        return OrientationVec.fromCoordinatesNd(orientationMatrix.T[0])
    
    def _computeBaitCompass(self) -> OrientationVec:
        posB = self._extractBaitPosition()
        posA = self._extractAgentPosition()
        
        # Create a bait compass
        return OrientationVec.fromCoordinatesNd(posB - posA)
    
    """
    Section related to logging info about simulated object
    """
    @_logged("Bait")
    def baitInfo(self):
        pos, _, _ = CustomAviary._getPosAndOrient(self.BAIT_ID, self.CLIENT)
        print(f"Position: {pos}")
        
    @_logged("Agent")
    def agentInfo(self):
        pos, rpy, orientationMatrix = CustomAviary._getPosAndOrient(self.DRONE_IDS[0], self.CLIENT)
        print(f"\tPosition: {pos}\n\tRoll Pitch Yaw: {rpy}\n\tOrientation matrix:\n{orientationMatrix}")

    @_logged("Obstacles")
    def obstacleInfo(self):
        for count, pillarData in enumerate(self.PILLAR_DATA):
            x, y, id = pillarData
            print(f"{count + 1}) X: {x} Y:{y} ID: {id}")
    
    """
    Section related to BaseAviary API and gym API
    """
    def _baitCompassObservationSpace(self):
        return spaces.Box(
            low=np.array([-1]),
            high=np.array([1]),
            dtype=np.float32
        )

    def _observationSpace(self):
        # sensorObservationSpace = self.obstacleSensor.observationSpace()
        # baitCompassObservationSpace = self._baitCompassObservationSpace()

        # return spaces.Box(
        #     low=np.concatenate((kinematicObservationSpace.low, baitCompassObservationSpace.low), axis=0),
        #     high=np.concatenate((kinematicObservationSpace.high, baitCompassObservationSpace.high), axis=0),
        #     dtype=np.float32
        # )
        # kinObsSpace = super()._observationSpace()
        # newLow = np.concatenate((kinObsSpace.low[:3], kinObsSpace.low[6:9]), axis=0)
        # newHigh = np.concatenate((kinObsSpace.high[:3], kinObsSpace.high[6:9]), axis=0)
        
        return super()._observationSpace()
    
    def _computeObs(self):
        state = self._getDroneStateVector(0)
        kinObservation = np.hstack(
            [
                state[0:3],   # xyz
                state[7:10],  # rpy
                state[10:13], # velocity
                state[13:16]  # angular velocity
            ]
        ).reshape(12,)
        
        posA = self._extractAgentPosition()
        posB = self._extractBaitPosition()
        
        kinObservation[0:3] = posA - posB

        return kinObservation
        # NOTE
        # sensorObservation = self.obstacleSensor.getReadyReadings()
        # baitCompass = np.array([self._extractAgentOrientationVector().correlateTo(self._computeBaitCompass())])
        # return np.concatenate((kinObservation, sensorObservation, baitCompass), axis=0)
    
    def _getTimeRewardMultiplier(self):
        # Reward mutation that comes from time spent on an episode
        # essentially the goal is, the longer the worse it is
        pass

    @_weighted(1.5)
    def _getBaitCompassRewardComponent(self):
        
        baitCompass = self._computeBaitCompass()
        agentOrientation = self._extractAgentOrientationVector()
        colinearParam = agentOrientation.correlateTo(baitCompass) # values in [-1, 1]
        
        return colinearParam
    
    @_weighted(1.0)
    def _getBaitDistanceRewardComponent(self):     
        posA = self._extractAgentPosition()
        posB = self._extractBaitPosition()
        
        distance = np.linalg.norm(posB[:-1] - posA[:-1])
        distanceParam = 0.0
        # Described on Desmos https://www.desmos.com/calculator/hwk0bpdl6y
        if (distance < CustomAviary.BAIT_SPHERE_RADIUS):
            # (0, 1.1]
            distanceParam = -200 * (distance - CustomAviary.BAIT_SPHERE_RADIUS) * (distance + CustomAviary.BAIT_SPHERE_RADIUS)
        elif (distance >= CustomAviary.BAIT_SPHERE_RADIUS):
            # [-inf, 0]
            distanceParam = -4 * (distance - CustomAviary.BAIT_SPHERE_RADIUS)*(distance + CustomAviary.BAIT_SPHERE_RADIUS)
        
        # z component penalty
        if (posA[2] - posB[2] > 2.5 * CustomAviary.BAIT_SPHERE_RADIUS):
            distanceParam -= 10 * (posA[2] - posB[2])
        
        if (posA[2] - posB[2] < CustomAviary.BAIT_SPHERE_RADIUS / 1.5):
            distanceParam -= 4

        return distanceParam
    
    def _getTotalBaitRewardComponent(self):
        # self._getBaitCompassRewardComponent()
        return self._getBaitDistanceRewardComponent()

    @_weighted(0.85)
    def _getObstacleProximityRewardComponent(self):
        proximityParam = 0
        # check critical proximity to obstacles, if any -> CustomAviary.ABSOLUTE_PENALTY
        # and compute reward based on self.obstacleSensor.lastRayReading done in one loop not to waste CPU cycles
        for dist in self.obstacleSensor.lastRayReading:
            if dist == -1 : continue
            
            if dist < self.criticalDistance:
                self.finito = True
                return CustomAviary.ABSOLUTE_PENALTY
            
            proximityParam += 1. / dist
        
        # squeezes proximity parameter to [-1, 1] range
        # proximityParam -> inf, return -> -1       this idenfifies very close proximity to obstacles
        # proximityParam -> 0, return -> 0          this idenfifies absence proximity to obstacles
        return ( 2. / (1 + np.exp(proximityParam / 10)) ) - 1

    def _computeReward(self):
        # At each reward call agent should obtain a reward within [-5, 5], where -10 is yielded for crashing or getting out of bounds
        # Boundaries are dictated by the forestProvider

        # compute distance with bait component
        baitDistComponent = self._getTotalBaitRewardComponent()

        # NOTE: it seems that agent may be unable to understand the correlation between position and punishment as position is not given to it as input
        # check withn boudaries, if outside -> -10
        posA = self._extractAgentPosition()
        posB = self._extractBaitPosition()
        # xA, yA, zA = posA[0], posA[1], posA[2]
        # xBoundary, yBoundary, zBoundary = self.forestProvider.getPoissonForrestGeometry()
        # xMin, xMax = xBoundary
        # yMin, yMax = yBoundary
        # zMin, zMax = zBoundary
        # if (
        #     (xA >= xMax or xA <= xMin) or
        #     (yA >= yMax or yA <= yMin) or
        #     (zA >= zMax or zA <= zMin)
        # ):
        #     self.finito = True
        #     return CustomAviary.ABSOLUTE_PENALTY
        
        # NOTE 
        # obstacleProximityComponent = self._getObstacleProximityRewardComponent()
        obstacleProximityComponent = 0
        totalReward = 0
        if (self.fDebug):
            self.rewardBuffer.append((baitDistComponent, obstacleProximityComponent))
        if (
            self.fStrictDeath and
            self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC and 
            np.linalg.norm(posA - posB) > CustomAviary.BAIT_SPHERE_RADIUS
        ):
            self.finito = True
            totalReward += CustomAviary.ABSOLUTE_PENALTY

        totalReward += baitDistComponent + obstacleProximityComponent
        return totalReward
    
    def rewardBufferInfo(self):
        assert self.fDebug
        self.rewardBuffer.info()

    def _computeDone(self):
        if (self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC):
            return True
        
        return self.finito
    
    def _computeInfo(self):
        return {"info": "Test run"}
    
    def reset(self):
        self.finito = False
        self._resetForest()
        self._resetPositionToRand()
        return super().reset()

