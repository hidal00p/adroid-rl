from enum import Enum

class NeuralNetConventions(Enum):
    MAIN_FIELD = ("nn", True)

    ARCH = ("netArch", True)
    ACTIVATION = ("activationFn", False)

class ObstacleSensorConventions(Enum):
    MAIN_FIELD = ("obst", True)

    VISION_ANGLE = ("visionAngle", False)
    N_SEG = ("nSegments", False)

class AviaryConventions(Enum):
    MAIN_FIELD = ("aviary", True)

    NUM_EPISODES = ("nEpisodes", False)
    EP_AV_N_STEPS = ("avEpisodeSteps", False)

    IS_STRICT_DEATH = ("isStrictDeath", False)
    BAIT_RESET_FREQ = ("baitResetFreq", False)
    EVAL_FREQ = ("evalFreq", False)


