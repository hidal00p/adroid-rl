import numpy as np
import pkg_resources
import pybullet as p

from gym_pybullet_drones.envs.single_agent_rl import HoverAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

class TrigConsts:
    PI = 4 * np.arctan(1)
    DEG2RAD = PI / 180
    RAD2DEG = 1 / DEG2RAD

class CustomAviary(HoverAviary):

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 forestGrid = None,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM):
                 """
                 CustomAviary class that generates a forest of obstacles to navigate through
                 """

                 self.PILLAR_DATA : list((float, float, int)) = [] # tuple of format (x_coord, y_coord, pybullet_id)
                 self.forestGrid = forestGrid

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

    def getNextCoordOnCircle(it: int, numPoints: int, x: float = 0, y: float = 0, r: float = 1) -> float:
        atomicArc : float = 360 / (numPoints - 1) # 360 degrees / number of unit arcs
        atomicArc = atomicArc * TrigConsts.DEG2RAD
        
        return (x + r * np.cos(atomicArc * it), y + r * np.sin(atomicArc * it))

    def _generatePillarTerrain(self):
        pass

    def _generatePillar(self, coords):
        x, y = coords
        id = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/column.urdf'),
            [x, y, .2],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT
            )
        
        self.PILLAR_DATA.append((x, y, id))

    def _addObstacles(self):
        if (self.forestGrid == None):
            # Generate a circular forest
            numPillars = 10
            for i in range(numPillars - 1):
                self._generatePillar(
                    CustomAviary.getNextCoordOnCircle(i, numPillars)
                )
        else:
            # Generate a forest from point grid
            for root in self.forestGrid:
                coords = (root[0], root[1])
                self._generatePillar(coords)