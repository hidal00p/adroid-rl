import argparse
import tests as t

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple testing environment for functionality and ideas.")
    parser.add_argument("--case", type=str, choices=t.TestCases.keys(), help="A single name of the test case to run.", required=True)
    parser.add_argument("--path", type=str, help="Provide a file path to the model to test", required=False)
    ARGS = parser.parse_args()

    testCase = vars(ARGS)["case"]
    # path = vars(ARGS)["path"]

    # if (testCase == "test-model" and path == None) or (testCase != "test-model" and path != None):
    #     raise RuntimeError("Incorrect usage")

    # if(testCase == "test-model"):
    #     t.TestModelPath = path

    t.TestCases[testCase]()
