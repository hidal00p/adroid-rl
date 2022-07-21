import numpy as np
import pkg_resources
import pybullet as p

from gym_pybullet_drones.envs.single_agent_rl import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from impl.utils.utils import ForestProvider, OrientationVec

class CustomAviary(HoverAviary):
    
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


    def _generatePillar(self, coords):
        x, y = coords
        id = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/column.urdf'),
            [x, y, .4 / 2],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT
            )
        
        self.PILLAR_DATA.append((x, y, id))
    
    def _addBait(self):
        """
        1. [x] Bait needs physical parameters - most likely will go with a simple cube, or a samurai XD 
        2. [x] Bait needs a position computed with respect to the forest's geometry
            2.1 [x] Bait will probably be positioned at the forest boundary + offset
        3. [x] Bait needs to be registered
        """
        x, y = self.forestProvider.baitCoordinates
        self.BAIT_ID = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/bait.urdf'),
            [x, y, .025 / 2],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT
        )

    def _getPosAndOrient(bodyId, clientId):
        pos, quat = p.getBasePositionAndOrientation(bodyId, clientId)
        rpy, orientationMatrix = p.getEulerFromQuaternion(quat), p.getMatrixFromQuaternion(quat)
        
        x = orientationMatrix[0:3]
        y = orientationMatrix[3:6]
        z = orientationMatrix[6:9]
        
        orientationMatrix = np.array([x, y, z]).T
        return (pos, rpy, orientationMatrix)

    def _addObstacles(self):
        # Add bait
        self._addBait()

        # Add forest
        for root in self.forestProvider.forestGrid:
            coords = (root[0], root[1])
            self._generatePillar(coords)

    def _extractBaitPosition(self):
        pos, _, _ = CustomAviary._getPosAndOrient(self.BAIT_ID, self.CLIENT)
        return np.array(pos)
    
    def _extractAgentPosition(self):
        pos, _, _ = CustomAviary._getPosAndOrient(self.DRONE_IDS[0], self.CLIENT)
        return np.array(pos)
    
    def computeBaitCompass(self) -> OrientationVec:
        posB = self._extractBaitPosition()
        posA = self._extractAgentPosition()
        
        # Create a bait compass
        return OrientationVec.fromCoordinatesNd(posB - posA)
    
    def extractBaitOrientationVector(self) -> OrientationVec:
        _, _, orientationMatrix = CustomAviary._getPosAndOrient(self.BAIT_ID, self.CLIENT)
        return OrientationVec.fromCoordinatesNd(orientationMatrix.T[0])

    def extractAgentOrientationVector(self) -> OrientationVec:
        _, _, orientationMatrix = CustomAviary._getPosAndOrient(self.DRONE_IDS[0], self.CLIENT)
        return OrientationVec.fromCoordinatesNd(orientationMatrix.T[0])

    @_logged("Bait")
    def baitInfo(self):
        pos, rpy, _ = CustomAviary._getPosAndOrient(self.BAIT_ID, self.CLIENT)
        print(f"Position: {pos} Orientation: {rpy}")
        
    @_logged("Agent")
    def agentInfo(self):
        pos, rpy, orientationMatrix = CustomAviary._getPosAndOrient(self.DRONE_IDS[0], self.CLIENT)
        print(f"\tPosition: {pos}\n\tRoll Pitch Yaw: {rpy}\n\tOrientation matrix:\n{orientationMatrix}")

    @_logged("Obstacles")
    def obstacleInfo(self):
        for count, pillarData in enumerate(self.PILLAR_DATA):
            x, y, id = pillarData
            print(f"{count + 1}) X: {x} Y:{y} ID: {id}")
    
