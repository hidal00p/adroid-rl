import numpy as np

from utils import ForestProvider as FP
from aviary.CustomAviary import CustomAviary as CA

def getEnv(fGui = False):
    # Define env and forest provider
    forestProvider = FP(fPoissonGrid=True, fDebug=True)
    env = CA(
        initial_xyzs=np.array([[0, 0, .15]]),
        gui=fGui,
        forestProvider=forestProvider
    )
    return env
