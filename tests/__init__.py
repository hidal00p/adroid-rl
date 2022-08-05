"""
Functional tests to check method correctness or validate ideas.
It is a simple script that fetches a particular function, finds its annontated value, and runs it if it is asked for.
Test cases a priori do not accept arguments. They are run as is, all the neccessary information should be defined within the test case.
"""

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
        obs, _, _, _ = env.step(np.array([.1, 0, 0, .0]))
        print(obs)
        i += 1
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
        _, reward, done, _ = env.step(np.array([.5, 0, 0, .5]))
        i += 1
        print(f"Reward at {i}: {reward} -> {done}")
        # if i % 250 == 0:
            # env.rewardBufferInfo()
        
        if done:
            env.reset()

@testCase("test-model")
def testTrainedModel():
    from stable_baselines3 import SAC
    from stable_baselines3 import PPO

    from aviary.utils import getEnv
    from agent.sensor import VisionParams
    import utils.file as uf
    
    env = getEnv(
        fGui=True,
        fDebug=False,
        visionParams=VisionParams(
            visionAngle=120,
            nSegments=121,
            range=.4
        )
    )

    modelFile = "best_model"
    # modelFile = "final"
    # modelFile = "inter"
    
    modelFolder = "(ppo-xy-diff_obs-strickt_death-simple_find)-relu-750000-120deg-121-384-256"
    
    # model = SAC.load(f"models/{modelFolder}/{modelFile}")
    model = PPO.load(f"models/{modelFolder}/{modelFile}")
    
    obs = env.reset()
    
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        if done:
            obs = env.reset()

@testCase("hello-world")
def testHelloWorld():
    print("Hello World")
