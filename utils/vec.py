import numpy as np

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

    """
    Functionally performs the same actions as `fromCoordinates`, however takes numpy array as argument directly
    """    
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