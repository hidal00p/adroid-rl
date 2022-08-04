from enum import Enum
import numpy as np
from gym import spaces
import pkg_resources
import pybullet as p

from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import BaseSingleAgentAviary, ObservationType
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from agent.sensor import ObstacleSensor, VisionParams

from utils import ForestProvider, OrientationVec

class ExtendedObservationType(Enum):
    LIDAR = "lidar" # Observation type to accomodate for obstacle awareness. Dimension is based on agent.sensor.ObstacleSensor.detectObstacles

class CustomAviary(BaseSingleAgentAviary):
    
    """
    Custom aviary inherits from BaseSingleAgentAviary
    """

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
    with ExtendedObservationType.LIDAR for obstacle awareness.
    CustomAviary takes responsibility for appending observation vector of ExtendedObservationType.LIDAR
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
                 visionParams: VisionParams=VisionParams()
                ):
                 """
                 CustomAviary class operates an agent, which attempts to navigate through a forest of obstacles.
                 """
                 
                 assert forestProvider != None, "Please provide a correctly configured forest"

                 self.PILLAR_DATA : list((float, float, int)) = [] # tuple of format (x_coord, y_coord, pybullet_id)
                 self.forestProvider = forestProvider
                 self.ObstacleSensor = ObstacleSensor(self, visionParams=visionParams)

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

    """
    Section related to spawning obstacles
    """
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

        # Add forest
        for root in self.forestProvider.forestGrid:
            self._generatePillar(tuple(root))
    
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
    
    def computeBaitCompass(self) -> OrientationVec:
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
    def _observationSpace(self):
        kinematicObservationSpace = super()._observationSpace()
        sensorObservationSpace = self.ObstacleSensor.observationSpace()

        return spaces.Box(
            low=np.concatenate((kinematicObservationSpace.low, sensorObservationSpace.low), axis=0),
            high=np.concatenate((kinematicObservationSpace.high, sensorObservationSpace.high), axis=0),
            dtype=np.float32
        )
    
    def _computeObs(self):
        kinObservation = super()._computeObs()
        sensorObservation = self.ObstacleSensor.getReadyReadings()
        
        return np.concatenate((kinObservation, sensorObservation), axis=0)
    
    def _clipAndNormalizeState(self,
                            state
                            ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                        normalized_pos_z,
                                        state[3:7],
                                        normalized_rp,
                                        normalized_y,
                                        normalized_vel_xy,
                                        normalized_vel_z,
                                        normalized_ang_vel,
                                        state[16:20]
                                        ]).reshape(20,)

        return norm_and_clipped
