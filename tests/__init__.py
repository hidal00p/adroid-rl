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
    
    obs = os.detectObstacles()

    for ob in obs:
        print(ob)

@testCase("hello-world")
def testHelloWorld():
    print("Hello World")
