import numpy as np

from utils import ForestProvider as FP
from aviary.CustomAviary import CustomAviary as CA

def getEnv(fGui = False, initial_xyzs = np.array([[0, 0, .15]]), initial_rpys = np.array([[0, 0, 0]])):
    # Define env and forest provider
    forestProvider = FP(fPoissonGrid=True, fDebug=True)
    env = CA(
        gui=fGui,
        forestProvider=forestProvider,
        initial_xyzs=initial_xyzs,
        initial_rpys=initial_rpys
    )
    return env
