import matplotlib.pyplot as plt
import math
import random
import numpy as np

# Init
def initPlt():
    plt.ion()

# Close
def closePlt():
    plt.ioff()
    plt.close()

# Stream
def rgbStream(rgb):
    print(len(rgb), len(rgb[0]))
    plt.imshow(rgb)

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

class TrigConsts:
    PI = 4 * np.arctan(1)
    DEG2RAD = PI / 180
    RAD2DEG = 1 / DEG2RAD

class ForestProvider:
    """
    Takes full responsibility of generating forest map.
    It provides client code with a map to be used.
    """

    def __init__(self, fPoissonGrid = False, fDebug = False):
        self.forestGrid = []
        self.baitComfortInterval = .05
        if fPoissonGrid:
            self._generatePoissonForest()
        else:
            self._generateCircularForest()
        
        assert len(self.forestGrid) != 0 and self.baitCoordinates != None, "Forest and/or bait configured incorrectly"
        if fDebug:
            print(f"[DEBUG INFO FOR {__class__}]:\nforestGrid: {self.forestGrid}\nbaitCoordinates: {self.baitCoordinates}")
        

    def _generatePoissonForest(self):
        size = 1.6
        sep = .25
        x_offset = .2
        y_offset = -size / 2
        
        p = Poisson2D(
            size = size, 
            sep = sep,
            offSetPair = (x_offset, y_offset),
            fDebug = False
        )

        self.forestGrid = p.generate().get()
        self.baitCoordinates = ((1 + self.baitComfortInterval) * (size + x_offset), 0)
    
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

class OrientationVec:
    """
    Computes orientation vector, and defines some operations on orientation vectors.
    Always contains coordinates of a unit vector, so it mormalizes the vector
    """

    def assertive(func):
        def _assertive(a, b):
            assert isinstance(b, OrientationVec)
            return func(a, b)
        
        return _assertive

    def fromCoordinatesNd(coords: np.ndarray):
        vec = OrientationVec()

        # Normalization
        n = np.linalg.norm(coords)
        vec.coords = 0 if n == 0 else coords / n

        return vec

    """
    Returns a unit vector from coordinates (X, Y, Z), which must be the the difference between X_A, Y_A, Z_A and X_B, Y_B, Z_B which define vector AB in R^3
    """
    def fromCoordinates(coords):
        vec = OrientationVec()
        x, y, z = coords
        
        # Edge cases
        x = 1 if x == None else x
        y = 0 if y == None else y
        z = 0 if z == None else z
        
        # Normalization
        _vec = np.array([x, y, z])
        n = np.linalg.norm(_vec)
        vec.coords = 0 if n == 0 else _vec / n
        
        return vec
        
    @assertive
    def correlateTo(a, b):
        return np.dot(a.coords, b.coords)

    @assertive
    def __add__(a, b):
        # Always normalizes the resultant direction
        return OrientationVec.fromCoordinatesNd(a.coords + b.coords)

    @assertive
    def __sub__(a, b):
        # Always normalizes the resultant direction
        return OrientationVec.fromCoordinatesNd(a.coords - b.coords)
    
    def __str__(self) -> str:
        return str(self.coords)
        