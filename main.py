from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import signal, sys

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
    importPath = "config.yml" if not IS_RESET else f"models/{PATH}/config.yml"
    traingSetting = [
        importConfig(importPath)
    ]
    
    for trainingConfig in traingSetting:
        train(trainingConfig)

def train(trainingConfig: TrainingConfig):
    print(
        "[INFO] START:\n"
        "==========================\n\n"
    )
    signature = trainingConfig.getSig()
    netArch,\
    activationFn,\
    visionAngle,\
    nSegments,\
    compressionParam,\
    simFreq,\
    avEpisodeSteps,\
    totalSteps,\
    collisionDistance,\
    isStrictBoundary,\
    isStrictDeath,\
    baitResetFreq,\
    evalFreq = trainingConfig.getConfig()
    
    activationName, activation  = activationFn
    fYaw = True
    if visionAngle == 360:
        fYaw = False

    sa_env_kwargs = dict(
        visionParams=VisionParams(
            visionAngle=visionAngle,
            nSegments=nSegments        
        ),
        forestProvider=ForestProvider(
            fPoissonGrid=True,
            fDebug=False
        ),
        freq=simFreq,
        baitResetFrequency=baitResetFreq,
        avEpisodeSteps=avEpisodeSteps,
        fStrictDeath=isStrictDeath,
        aggregate_phy_steps=5,
        compressionParam=compressionParam,
        fStrictBoundary=isStrictBoundary,
        criticalDistance=collisionDistance,
        fYaw=fYaw,
        fTwoDimAction=True
    )

    trainEnv = make_vec_env(
        CustomAviary,
        env_kwargs=sa_env_kwargs,
        n_envs=6,
        seed=0
    )

    model_kwargs = dict(
        activation_fn=activation,
        net_arch=netArch
    )

    pathRoot = PATH if not IS_RESET else f"{PATH}-RESET"
    loadPath = f"models/{PATH}/{uf.getPathFromModelParams(constructSigStr(signature), netArch, visionAngle, nSegments, totalSteps, activationName)}"
    savePath = f"models/{pathRoot}/{uf.getPathFromModelParams(constructSigStr(signature), netArch, visionAngle, nSegments, totalSteps, activationName)}"
    assert IS_RESET == (loadPath != savePath), "Somethig is wrong with reseting"
    
    finalModelFile = "final.zip"
    interModelFile = "inter.zip"

    algo = SAC
    policy = SACPolicy
    entropyCoef = 0
    if signature["algo"] == "ppo":
        algo = PPO
        policy = ActorCriticPolicy
        entropyCoef = 0.01

    model = algo(
        policy,
        trainEnv,
        ent_coef=entropyCoef,
        policy_kwargs=model_kwargs,
        verbose=1
    )
    
    if IS_RESET:
        algo.load(f"{loadPath}/best_model")

    eval_env = make_vec_env(
        CustomAviary,
        env_kwargs=sa_env_kwargs,
        n_envs=1,
        seed=0
    )

    eval_callback = EvalCallback(
        eval_env,
        verbose=1,
        best_model_save_path=savePath,
        eval_freq=evalFreq,
        deterministic=True,
        render=False
    )
    
    # Define SIGINT handler
    def sigintHandler(sig, frame):
        model.save(f"{savePath}/{interModelFile}")

        trainEnv.close()
        print("Shutting down gracefully")
        sys.exit(0)
    
    # Register signal handler
    signal.signal(signal.SIGINT, sigintHandler)
    signal.signal(signal.SIGTERM, sigintHandler)
    
    try:
        model.learn(
            total_timesteps=totalSteps,
            callback=eval_callback
        )
        print("Finished learning")
    except:
        print("EXCEPTION CAUGHT")
        exit(0)
    
    print("Saving the final model...")
    model.save(f"{savePath}/{finalModelFile}")
    print(
        "\n\n[INFO] DONE:\n"
        "==========================\n\n"
    )

if __name__ == "__main__":
    PATH = sys.argv[1]
    IS_RESET = False
    
    if len(sys.argv) > 2:
        IS_RESET = True

    run()