"""
Functional tests to check method correctness or validate ideas.
It is a simple script that fetches a particular function, finds its annontated value, and runs it if it is asked for.
Test cases a priori do not accept arguments. They are run as is, all the neccessary information should be defined within the test case.
"""

from aviary.CustomAviary import CustomAviary


TestCases = {}
TestModelPath = ""

def testCase(testName = None):
    assert testName != None, "Please give your test a unique name"
    def _testable(func):
        if testName not in TestCases.keys():
            TestCases[testName] = func
        else:
            raise RuntimeError(
                f"Function _{func.__name__}_ cannot be registered under case name _{testName}_ as it is already in use.\n"
                f"A function _{TestCases[testName].__name__}_ was previously encountered among test case list.\n"
                "Please provide a unique name and re-run the program!"
                )
        
        return func

    return _testable

# hook1
# ============Test model main================

@testCase("test-model")
def testTrainedModel():
    from stable_baselines3 import SAC
    from stable_baselines3 import PPO

    from aviary.utils import getEnvFromConfig
    
    caseFolder = "sac-correct-strict-1"
    modelFolder = "sac-xyz_diff-z_lim-strict_death-kin_obs-vel-strict_correct-relu-1800000-120deg-121-400-300"
    modelFile = "best_model"
    # modelFile = "final"
    # modelFile = "inter"
    
    modelPath = f"models/{caseFolder}/{modelFolder}/{modelFile}"
    configPath = f"models/{caseFolder}/config.yml"

    env, sig = getEnvFromConfig(path=configPath)

    algo = sig["algo"]
    if algo == "ppo":
        model = PPO.load(modelPath)
    elif algo == "sac":
        model = SAC.load(modelPath)
    else:
        raise NotImplementedError()
    
    obs = env.reset()
    
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        print(reward)
        # if done:
        #     obs = env.reset()

# ============================================


@testCase("env-creation")
def testEnvCreation():
    import aviary.utils as au
    assert au.getEnv() != None

@testCase("agent-orientation")
def testAgentOrientation():
    from agent.sensor import VisionParams
    import aviary.utils as au

    env = au.getEnv(
        fGui=False, visionParams=VisionParams(nSegments=12)
    )

@testCase("calculate-ray-pencil")
def testRayPencilCalculation():
    from agent.sensor import ObstacleSensor
    import aviary.utils as au

    os = ObstacleSensor(
        env = au.getEnv(fGui=False)
    )

    print(os._calculateRayPencil())

@testCase("obstacle-detection")
def testObstacleDetection():
    from agent.sensor import ObstacleSensor
    import aviary.utils as au

    os = ObstacleSensor(
        env = au.getEnv(fGui=False)
    )
    
    obs = os._detectObstacles()

    for ob in obs:
        print(ob)
    
@testCase("custom-obs-type")
def testCustomObservationTypePolymorphism():
    from aviary.CustomAviary import ExtendedObservationType
    for obsType in ExtendedObservationType:
        print(obsType.value)

@testCase("obstacle-sensor-obs-space")
def testObstacleSensorObservationSpace():
    from agent.sensor import VisionParams
    import aviary.utils as au

    env = au.getEnv(
        fGui=False, visionParams=VisionParams(nSegments=12)
    )

    print(env.obstacleSensor.observationSpace())

@testCase("custom-aviary-obs-space")
def testCustomAviaryObservationSpace():
    from agent.sensor import VisionParams
    import aviary.utils as au

    env = au.getEnv(
        fGui=False, visionParams=VisionParams(nSegments=12)
    )
    
    print(env._observationSpace())

@testCase("compute-custom-obs")
def testCustomAviaryObservationSpace():
    import numpy as np
    from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType
    from agent.sensor import VisionParams
    import aviary.utils as au

    env = au.getEnv(
        fGui=True, 
        visionParams=VisionParams(nSegments=33, visionAngle=60, range=.35),
        initial_rpys=np.array([[0, 0, -np.pi / 2]])
    )

    i = 0
    while True:
        obs, _, done, _ = env.step(np.array([.1, 0.1, 0, .0]))
        print(obs)
        i += 1
        if done:
            env.reset()
        # if i % 250 == 0:
        #     print(f"LOG[{i}]:\n"
        #         f"{np.flip(env.obstacleSensor.getReadyReadings().reshape((env.obstacleSensor.visionParams.nSegments, 1)), axis=0)}\n"
        #     )

@testCase("compute-reward")
def testRewardComputation():
    from agent.sensor import VisionParams
    import aviary.utils as au
    import  numpy as np

    env = au.getEnv(
        fGui=True,
        fDebug=True,
        visionParams=VisionParams(nSegments=121, visionAngle=120),
    )

    i = 0
    while True:
        _, reward, done, _ = env.step(np.array([.0, .0, .1, .0]))
        i += 1
        print(f"Reward at {i}: {reward} -> {done}")
        # if i % 250 == 0:
            # env.rewardBufferInfo()
        
        # if done:
        #     env.reset()

@testCase("hello-world")
def testHelloWorld():
    print("Hello World")

@testCase("enums")
def testEnums():
    from signature import NeuralNetConventions
    print(
        f"{NeuralNetConventions.MAIN_FIELD}\n"
        f"{NeuralNetConventions}\n"
        f"{[sig for sig in NeuralNetConventions].pop(0)}\n"
    )

@testCase("kwargs")
def testKwargs():
    def f(a):
        print(a)
    
    kw = {
        "a":"b",
    }
    
    f(**kw)

@testCase("yaml-load")
def testLoadYaml():
    from utils.config import getConfig
    runConfig = getConfig("config.yml")
    print(runConfig)

@testCase("import-config")
def testConfigImporting():
    from utils.config import importConfig
    try:
        print(importConfig("config"))
        assert False
    except:
        assert True
    
    print(importConfig("config.yml"))

@testCase("train-conf-from-conf-file")
def testConfigImporting():
    from utils.config import importConfig
    from aviary.train import TrainingConfig
    
    tc : TrainingConfig = importConfig()

    print(tc.getConfig())
    
@testCase("construct-aviary-from-config")
def testConfigImporting():
    import numpy as np
    from aviary.CustomAviary import CustomAviary
    from aviary.train import TrainingConfig
    from agent.sensor import VisionParams
    
    from utils.config import importConfig
    from utils.terrain import ForestProvider
    
    
    tc : TrainingConfig = importConfig()

    netArch, activationFn, visionAngle, nSegments, simFreq, avEpisodeSteps, totalSteps, isStrictDeath, baitResetFreq, evalFreq = tc.getConfig()

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
        aggregate_phy_steps=5,
        gui=True
    )

    env = CustomAviary(**sa_env_kwargs)
    
    while True:
        _, reward, done, _ = env.step(np.array([.0, 0.0, .2, .2]))
        # print(reward)
        if done:
            print(reward)
            env.close()
            exit(0)
    
