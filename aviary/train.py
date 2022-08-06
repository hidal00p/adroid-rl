# Number of episodes ->
# Av number of steps in an episode -> 
# Frequency of bait position refresh ->
# Frequency of env evaluation ->
import torch

ActiovationRegisty = {
    "relu": ("relu", torch.nn.ReLU),
    "leakyrelu": ("leakyrelu", torch.nn.LeakyReLU),
}

class TrainingConfig():
    def __init__(
        self,
        signature={"algo": "sac"},
        # NN arch args
        netArch=[],
        activationFn="relu",
        
        # Obstacle sensor args
        visionAngle=120,
        nSegments=121,
        
        # Aviary params
        nEpisodes=5_000,
        avEpisodeSteps=1_000,
        
        isStrictDeath=True,
        baitResetFreq=10,
        evalFreq=2_000
    ):  
        self.signature = signature

        self.netArch = netArch
        self.activationFn = ActiovationRegisty[activationFn]

        self.visionAngle = visionAngle
        self.nSegments = nSegments
        
        self.nEpisodes = nEpisodes
        self.avEpisodeSteps = avEpisodeSteps
        self.isStrictDeath = isStrictDeath
        self.baitResetFreq = baitResetFreq
        self.evalFreq = evalFreq
        self.totalSteps = self.nEpisodes * self.avEpisodeSteps
    
    def getSig(self):
        return self.signature
    
    def getConfig(self):
        return (self.netArch, self.visionAngle, self.nSegments, self.evalFreq, self.totalSteps, self.activationFn)
    
    def construct(**kwargs):
        return TrainingConfig(**kwargs)