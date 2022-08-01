import numpy as np
import pkg_resources
import pybullet as p

from gym_pybullet_drones.envs.single_agent_rl import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from utils import ForestProvider, OrientationVec

class CustomAviary(HoverAviary):
    
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
                 obs: ObservationType=ObservationType.RGB,
                 act: ActionType=ActionType.RPM):
                 """
                 CustomAviary class operates an agent, which attempts to navigate through a forest of obstacles.
                 """
                 
                 assert forestProvider != None, "Please provide a correctly configured forest"

                 self.PILLAR_DATA : list((float, float, int)) = [] # tuple of format (x_coord, y_coord, pybullet_id)
                 self.forestProvider = forestProvider

                 super().__init__(
                    drone_model=drone_model,
                    initial_xyzs=initial_xyzs,
                    initial_rpys=initial_rpys,
                    physics=physics,
                    freq=freq,
                    aggregate_phy_steps=aggregate_phy_steps,
                    gui=gui,
                    record=record,
                    obs=obs,
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
    
