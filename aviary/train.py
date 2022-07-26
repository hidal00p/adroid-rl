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

        compressionParam=None,
        
        # Aviary params
        nEpisodes=5_000,
        simFreq=240, # Hz, updates per second
        avEpisodeSteps=1_000,
        
        collisionDistance=.1,
        isStrictBoundary=True,
        isStrictDeath=True,
        baitResetFreq=10,
        evalFreq=2_000
    ):  
        self.signature = signature

        self.netArch = netArch
        self.activationFn = ActiovationRegisty[activationFn]

        self.visionAngle = visionAngle
        self.nSegments = nSegments

        self.compressionParam = compressionParam
        
        self.nEpisodes = nEpisodes
        self.simFreq = simFreq
        self.avEpisodeSteps = avEpisodeSteps
        self.totalSteps = self.nEpisodes * self.avEpisodeSteps
        
        self.collisionDistance = collisionDistance
        self.isStrictBoundary = isStrictBoundary
        self.isStrictDeath = isStrictDeath
        self.baitResetFreq = baitResetFreq
        self.evalFreq = evalFreq
    
    def getSig(self):
        return self.signature
    
    def getConfig(self):
        return (
            self.netArch,
            self.activationFn,
            
            self.visionAngle,
            self.nSegments,
            
            self.compressionParam,

            self.simFreq,
            self.avEpisodeSteps,
            self.totalSteps,
            
            self.collisionDistance,
            self.isStrictBoundary,
            self.isStrictDeath,
            self.baitResetFreq,
            self.evalFreq
        )
    
    def construct(**kwargs):
        return TrainingConfig(**kwargs)