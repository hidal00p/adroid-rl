from enum import Enum

class SignatureConventions(Enum):
    MAIN_FIELD = ("sig", True)

    ALGORITHM = ("algo", True)
    REWARD = ("reward", False)
    PENALTY = ("penalty", False)
    OBSERVATION = ("obs", False)
    ACTION = ("act", False)
    OTHERS = ("others", False)
