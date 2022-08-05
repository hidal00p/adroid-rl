def getPathFromModelParams(pathSig, nnArch, visionAngle, nSegments, totalTimeSteps, activationName):
    path = f"{pathSig}-{activationName}-{totalTimeSteps}-{visionAngle}deg-{nSegments}"
    for dim in nnArch:
        path += f"-{dim}"
    return path

def getModelParmsFromPath(path):
    pass