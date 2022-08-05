import matplotlib.pyplot as plt
import math, random
import numpy as np

from utils.trig import TrigConsts

class Poisson2D:
    def __init__(self, size = 1, sep = 0.05, fDebug = False, offSetPair = None ):
        self.offSetPair = offSetPair
        # Assure correctness of the offSetPair parameter
        if self.offSetPair:
            _x, _y = self.offSetPair
            assert _x != None and _y != None, "Please provide a full offSetPair tuple. If you would like to shift only one coordinate please set the other to 0.0"
        
        self.fDebug = fDebug
        self.__size = size
        self.__sep = sep
        self.__dx = sep / math.sqrt(2)
        self.__ngrid = math.floor(self.__size / self.__dx)

        self.grid = np.empty((self.__ngrid * self.__ngrid, 2), dtype=float)
        self.grid.fill(-1)
        self.active = []
        self.index_all = []

    def generate(self):
        # initial starting from the center
        x0 = self.__size * 0.5
        y0 = self.__size * 0.5
        i_row = math.floor(x0 / self.__dx)
        i_col = math.floor(y0 / self.__dx)
        index = self.__coord2Index(i_row, i_col)
        self.grid[index,0] = x0
        self.grid[index,1] = y0
        self.index_all.append(index)
        self.active.append(index)

        # start random sampling
        while len(self.active) > 0:
            self.__newPts()
        
        if self.offSetPair:
            self._shiftPoints()

        return self

    def __newPts(self):
        # get a random active points
        randIndex = math.floor(random.uniform(0, len(self.active)))
        refIndex = self.active[randIndex]
        ref_pos = self.grid[refIndex]
        flag_found = False
        # try to sample at most 30 points
        for i_trial in range(30):
            # trail position
            radius = random.uniform(self.__sep, 2*self.__sep)
            angle = random.uniform(0, 2*math.pi)
            trial_pos = ref_pos + np.array([radius * math.cos(angle), radius * math.sin(angle)], dtype=float)
            # check trial position
            trial_row = math.floor(trial_pos[0] / self.__dx)
            trial_col = math.floor(trial_pos[1] / self.__dx)
            trial_index = self.__coord2Index(trial_row, trial_col)

            if not self.__isInside(trial_row, trial_col):
                continue
            # check if inside target shape
            # if not self.__isInRectangle(trial_pos[0], trial_pos[1]):
            #     continue
            # if not self.__isInCircle(trial_pos[0], trial_pos[1]):
            #     continue
            if self.__isOccupied(trial_index):
                continue
            
            # check the neighbor of the trial position
            flag_good_pos = self.__isNeighborGood(trial_row, trial_col, trial_pos)

            if flag_good_pos:
                self.grid[trial_index] = trial_pos
                self.index_all.append(trial_index)
                self.active.append(trial_index)
                flag_found = True
        # if no such point can be found, remove the reference point
        if not flag_found:
            self.active.pop(randIndex)
    
    def __coord2Index(self, i, j):
        return i + j * self.__ngrid

    def __isOccupied(self, index):
        if self.grid[index,0] == -1 and self.grid[index,1] == -1:
            return False
        else:
            return True

    def __isInside(self, row, col):
        if row < self.__ngrid and row >= 0 and col < self.__ngrid and col >= 0:
            return True
        else:
            return False        
    
    def __isNeighborGood(self, row, col, pos):
        for i in range(-1,2):
            for j in range(-1,2):
                if self.__isInside(row+i, col+j):
                    check_index = self.__coord2Index(row+i, col+j)
                    if self.__isOccupied(check_index):
                        dist = np.linalg.norm(pos - self.grid[check_index])
                        if dist < self.__sep:
                            return False
        return True

    def __isInRectangle(self, x, y):
        b_length = 1.0
        b_width = 0.5
        if x >= 0 and x <= b_length and y >= 0 and y <= b_width:
            return True
        else:
            return False

    def __isInCircle(self, x, y):
        b_radius = 0.5
        dist = math.sqrt((x-0.5)**2 + (y-0.5)**2)
        if dist <= b_radius:
            return True
        else:
            return False
    
    def _translateToRectangularRegion():
        pass

    def _shiftPoints(self):
        """ Points indexed in self.index_all will be shifted from (0, 0) origin to (offset_x, offset_y)"""
        assert self.offSetPair != None, "Offset pair cannot be None when calling to _shiftPoints()"
        
        offset_x, offset_y = self.offSetPair
        for index in self.index_all:
            self.grid[index, 0] += offset_x
            self.grid[index, 1] += offset_y

    def draw(self):
        ax = plt.figure().add_subplot(111)
        ax.set_aspect('equal')
        for index in self.index_all:
            ax.plot(self.grid[index,0], self.grid[index,1], marker='.',markersize=2, color='k')
        print(f"[DEBUG INFO]:\n\tlen: {len(self.index_all)}\n\tsize: {self.__size}\n\tsep: {self.__sep}\n\tngrid: {self.__ngrid}")
        plt.xlim(0, self.__size)
        plt.ylim(0, self.__size)
        plt.show()

    def get(self):
        if self.fDebug:
            self.draw()

        coords = []
        for index in self.index_all:
            coords.append((self.grid[index,0], self.grid[index,1]))
        
        return coords

class ForestProvider:
    """
    Takes full responsibility of generating forest map.
    It provides client code with a map to be used.
    """

    def __init__(
        self, 
        fPoissonGrid = False, 
        fDebug = False,
        # PoissonForestParams
        forestSize = 1.6, 
        densityParameter = .3, 
        x_offset = .2
    ):
        self.forestGrid = []
        self.baitComfortInterval = .05
        self.fPoissonGrid = fPoissonGrid

        if fPoissonGrid:
            self.forestSize = forestSize
            self.densityParameter = densityParameter
            self.x_offset = x_offset

            self._generatePoissonForest()
        else:
            self._generateCircularForest()
        
        assert len(self.forestGrid) != 0 and self.baitCoordinates != None, "Forest and/or bait configured incorrectly"
        if fDebug:
            print(f"[DEBUG INFO FOR {__class__}]:\nforestGrid: {self.forestGrid}\nbaitCoordinates: {self.baitCoordinates}")
    
    def resetBaitPosition(self):
        self.baitCoordinates = (
            np.random.uniform(self.x_offset, self.x_offset + self.forestSize), 
            np.random.uniform(-0.5*self.forestSize, 0.5*self.forestSize)
        )

    def getPoissonForrestGeometry(self):
        if not self.fPoissonGrid:
            raise RuntimeError("Accessing method that is only available for Poisson grid")
        
        # Returns boundary values ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        baitX, _ = self.baitCoordinates
        return ((self.x_offset - .3, (1 + self.baitComfortInterval) * baitX), (-self.forestSize/2, self.forestSize/2), (0.06, .45))

    def _generatePoissonForest(self):
        y_offset = -self.forestSize / 2
        
        p = Poisson2D(
            size = self.forestSize, 
            sep = self.densityParameter,
            offSetPair = (self.x_offset, y_offset),
            fDebug = False
        )

        self.forestGrid = p.generate().get()
        self.baitCoordinates = ((1 + self.baitComfortInterval) * (self.forestSize + self.x_offset), 0)
    
    def _generateCircularForest(self):
        numPillars = 10
        radius = 1
        x_c = 0.0
        y_c = 0.0

        self.forestGrid = ForestProvider.getCoordsOnCircle(
            numPillars = numPillars,
            x = x_c, 
            y = y_c,
            r = radius)
        
        self.baitCoordinates = ((1 + self.baitComfortInterval) * radius, 0)

    def getCoordsOnCircle(numPoints: int = None, x: float = 0, y: float = 0, r: float = 1) -> float:
        assert numPoints != None, "Please provide an unisigned integer number of poinst on the circle"
        
        atomicArc : float = 360 / (numPoints - 1) # 360 degrees / number of unit arcs
        atomicArc = atomicArc * TrigConsts.DEG2RAD
        
        coords = []
        for point in range(numPoints - 1):
            coords.append((x + r * np.cos(atomicArc * point), y + r * np.sin(atomicArc * point)))
        
        return coords