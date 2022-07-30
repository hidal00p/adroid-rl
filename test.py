import argparse
import tests as t

def test(testName):
    t.TestCases[testName]()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple testing environment for functionality and ideas.")
    parser.add_argument("--case", type=str, choices=t.TestCases.keys(), help="A single name of the test case to run.", required=True)
    ARGS = parser.parse_args()

    test(vars(ARGS)["case"])