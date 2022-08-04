"""
Functional tests to check method correctness or validate ideas.
It is a simple script that fetches a particular function, finds its annontated value, and runs it if it is asked for.
Test cases a priori do not accept arguments. They are run as is, all the neccessary information should be defined within the test case.
"""

TestCases = {}

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
    from agent.sensor import VisionParams
    import aviary.utils as au

    env = au.getEnv(
        fGui=True, 
        visionParams=VisionParams(nSegments=33, visionAngle=60, range=.35)
    )

    i = 0
    while True:
        env.step(np.zeros((4,)))
        i += 1
        if i % 250 == 0:
            print(f"LOG[{i}]:\n"
                f"{np.flip(env.obstacleSensor.getReadyReadings().reshape((env.obstacleSensor.visionParams.nSegments, 2)), axis=0)}\n"
            )

@testCase("compute-reward")
def testRewardComputation():
    from agent.sensor import VisionParams
    import aviary.utils as au
    import  numpy as np

    env = au.getEnv(
        fGui=True,
        fDebug=True,
        visionParams=VisionParams(nSegments=121, visionAngle=120),
        initial_xyzs=np.array([[1.85, 0, .15]])
        # initial_rpys=np.array([[0, 0, 0]])
    )

    i = 0
    while True:
        _, reward, done, _ = env.step(np.zeros((4,)))
        i += 1
        if i % 250 == 0:
            print(f"Reward at {i}: {reward} -> {done}")
            env.rewardBufferInfo()
        
        if done:
            env.reset()

@testCase("hello-world")
def testHelloWorld():
    print("Hello World")
