from typing import Tuple
import numpy as np
from agent.sensor import VisionParams
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType

from utils import ForestProvider as FP
from utils.config import importConfig
from aviary.CustomAviary import CustomAviary as CA

def getEnv(
    fGui = False,
    fDebug = False,
    initial_xyzs = np.array([[0, 0, .15]]), 
    initial_rpys = np.array([[0, 0, 0]]),
    visionParams=VisionParams(),
    actionType=ActionType.VEL
    ):
    # Define env and forest provider
    forestProvider = FP(fPoissonGrid=True, fDebug=True)
    env = CA(
        gui=fGui,
        fDebug=fDebug,
        forestProvider=forestProvider,
        initial_xyzs=initial_xyzs,
        initial_rpys=initial_rpys,
        visionParams=visionParams,
        act=actionType
    )
    return env

def getEnvFromConfig(path="config.yml", gui=True) -> Tuple[CA, dict]:
    config = importConfig(path)
    netArch, activationFn, visionAngle, nSegments, simFreq, avEpisodeSteps, totalSteps, isStrictDeath, baitResetFreq, evalFreq = config.getConfig()

    sa_env_kwargs = dict(
        gui=gui,
        visionParams=VisionParams(
            visionAngle=visionAngle,
            nSegments=nSegments,
            range=.45
        ),
        forestProvider=FP(
            fPoissonGrid=True,
            fDebug=False
        ),
        freq=simFreq,
        baitResetFrequency=baitResetFreq,
        avEpisodeSteps=avEpisodeSteps,
        fStrictDeath=isStrictDeath,
        aggregate_phy_steps=5
    )

    return CA(**sa_env_kwargs), config.getSig()