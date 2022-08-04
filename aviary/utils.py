import numpy as np
from agent.sensor import VisionParams

from utils import ForestProvider as FP
from aviary.CustomAviary import CustomAviary as CA

def getEnv(
    fGui = False, 
    initial_xyzs = np.array([[0, 0, .15]]), 
    initial_rpys = np.array([[0, 0, 0]]),
    visionParams=VisionParams()
    ):
    # Define env and forest provider
    forestProvider = FP(fPoissonGrid=True, fDebug=True)
    env = CA(
        gui=fGui,
        forestProvider=forestProvider,
        initial_xyzs=initial_xyzs,
        initial_rpys=initial_rpys,
        visionParams=visionParams
    )
    return env
