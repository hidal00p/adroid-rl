from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import signal, sys
import torch

from aviary.CustomAviary import CustomAviary
from aviary.train import TrainingConfig
from agent.sensor import VisionParams
from signature.utils import constructSigStr
from utils import ForestProvider
from utils.config import importConfig
import utils.file as uf

# Serves to help identify between models run according to separate ideas
# it also helps with marking processes

def run():
    traingSetting = [
        importConfig()
    ]

    for trainingConfig in traingSetting:
        train(trainingConfig)

def train(trainingConfig: TrainingConfig):
    print(
        "[INFO] START:\n"
        "==========================\n\n"
    )
    signature = trainingConfig.getSig()
    netArch, activationFn, visionAngle, nSegments, simFreq, avEpisodeSteps, totalSteps, isStrictDeath, baitResetFreq, evalFreq = trainingConfig.getConfig()
    
    activationName, activation  = activationFn

    sa_env_kwargs = dict(
        visionParams=VisionParams(
            visionAngle=visionAngle,
            nSegments=nSegments,
            range=.45
        ),
        forestProvider=ForestProvider(
            fPoissonGrid=True,
            fDebug=False
        ),
        freq=simFreq,
        baitResetFrequency=baitResetFreq,
        avEpisodeSteps=avEpisodeSteps,
        fStrictDeath=isStrictDeath,
        aggregate_phy_steps=5
    )

    trainEnv = make_vec_env(
        CustomAviary,
        env_kwargs=sa_env_kwargs,
        n_envs=2,
        seed=0
    )

    model_kwargs = dict(
        activation_fn=activation,
        net_arch=netArch
    )

    algo = SAC
    policy = SACPolicy
    if signature["algo"] == "ppo":
        algo = PPO
        policy = ActorCriticPolicy

    model = algo(
        policy,
        trainEnv,
        ent_coef=0.01,
        policy_kwargs=model_kwargs,
        verbose=1
    )
    
    eval_env = make_vec_env(
        CustomAviary,
        env_kwargs=sa_env_kwargs,
        n_envs=1,
        seed=0
    )

    path = f"models/{PATH}/{uf.getPathFromModelParams(constructSigStr(signature), netArch, visionAngle, nSegments, totalSteps, activationName)}"
    finalModelFile = "final.zip"
    interModelFile = "inter.zip"
    eval_callback = EvalCallback(
        eval_env,
        verbose=1,
        best_model_save_path=path,
        eval_freq=evalFreq,
        deterministic=True,
        render=False
    )
    
    # Define SIGINT handler
    def sigintHandler(sig, frame):
        model.save(f"{path}/{interModelFile}")

        trainEnv.close()
        print("Shutting down gracefully")
        sys.exit(0)
    
    # Register signal handler
    signal.signal(signal.SIGINT, sigintHandler)
    signal.signal(signal.SIGTERM, sigintHandler)

    model.learn(
        total_timesteps=totalSteps,
        callback=eval_callback
    )
    
    model.save(f"{path}/{finalModelFile}")
    print(
        "\n\n[INFO] DONE:\n"
        "==========================\n\n"
    )

if __name__ == "__main__":
    PATH = sys.argv[1]
    run()