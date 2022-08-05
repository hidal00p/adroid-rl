from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
import signal, sys
import torch

from aviary.CustomAviary import CustomAviary
from agent.sensor import VisionParams
from utils import ForestProvider
import utils.file as uf

# Serves to help identify between models run according to separate ideas
# it also helps with marking processes
SIGNATURE = "regular"

class TrainingConfig():
    def __init__(
        self,
        netArch,
        visionAngle,
        nSegments,
        evalFreq,
        totalSteps,
        activationFn
    ):
        self.netArch = netArch
        self.visionAngle = visionAngle
        self.nSegments = nSegments
        self.evalFreq = evalFreq
        self.totalSteps = totalSteps
        self.activationFn = activationFn
    
    def getConfig(self):
        return (self.netArch, self.visionAngle, self.nSegments, self.evalFreq, self.totalSteps, self.activationFn)

def run():
    traingSetting = [
        TrainingConfig(
            netArch=[448, 320, 64],
            visionAngle=120,
            nSegments=121,
            evalFreq=1000,
            totalSteps=200_000,
            activationFn=("relu", torch.nn.ReLU)
        )
    ]
    for trainingConfig in traingSetting:
        trainSac(trainingConfig)

def trainSac(trainingConfig: TrainingConfig):
    print(
        "[INFO] START:\n"
        "==========================\n\n"
    )
    netArch, visionAngle, nSegments, evalFreq, totalSteps, activationFn = trainingConfig.getConfig()
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
        aggregate_phy_steps = 5
    )

    trainEnv = make_vec_env(
        CustomAviary,
        env_kwargs=sa_env_kwargs,
        n_envs=1,
        seed = 0
    )

    model_kwargs = dict(
        activation_fn=activation,
        net_arch=netArch
    )

    model = SAC(
        SACPolicy,
        trainEnv,
        policy_kwargs=model_kwargs,
        verbose=1
    )
    
    eval_env = make_vec_env(
        CustomAviary,
        env_kwargs=sa_env_kwargs,
        n_envs=1,
        seed = 0
    )

    global SIGNATURE
    path = f"models/{uf.getPathFromModelParams(SIGNATURE, netArch, visionAngle, nSegments, totalSteps, activationName)}"
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
    SIGNATURE = sys.argv[1]
    run()