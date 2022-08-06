import yaml
from pyparsing import ParseException
from plistlib import InvalidFileException

from signature import *
from aviary.train import TrainingConfig

"""
Helper to convert config.yml -> TrainingConfig runtime structure
"""

def checkExistenceAndReq(trait: tuple, config : dict):
    traitName, fTraitReq = trait
    if (fTraitReq and traitName not in config.keys()):
        raise ParseException(
            "Incorrectly configured YAML. "
            f"Trait {traitName} is required but was not found"
        )
    
    if traitName in config.keys():
        return traitName, config[traitName]
    else:
        return None, None

def parse(config : dict, convention: ConventionType):
    convStack = [c for c in convention]
    mainName, mainReq = convStack.pop(0).value

    # Purposefully separating MAIN_FIELD from others
    if (mainReq and not mainName in config.keys()):
        raise ParseException(
            "Incorrectly configured YAML. "
            f"Main field {mainName} is required but was not found"
        )
    
    testConfigKwargs = {}

    for conv in convStack:
        traitName, traitVal = checkExistenceAndReq(conv.value, config[mainName])
        if traitName != None and traitVal != None:
            testConfigKwargs[traitName] = traitVal
    
    return testConfigKwargs

def getConfig(fileName = "config.yml"):
    with open(fileName, "r") as config:
        runConfig = yaml.safe_load(config)
    
    return runConfig

def importConfig(configFileName = "config.yml") -> TrainingConfig:
    if ".yml" not in configFileName:
        raise InvalidFileException("Please provid a valid YAML comfig file")
    
    runConfig = getConfig(configFileName)
    
    parseStack = [SignatureConventions, NeuralNetConventions, ObstacleSensorConventions, AviaryConventions]
    
    if not type(parseStack[0]) == type(SignatureConventions):
        raise TypeError("First convention in parsing stack has to be SignatureConventions")
    
        # Get TestConfig kwargs and construct extended signature
    testConfigKwargs = {
        "signature": parse(runConfig, parseStack.pop(0))
    }
    
    for conv in parseStack:
        testConfigKwargs = {**testConfigKwargs, **parse(runConfig, conv)}
    
    return TrainingConfig.construct(**testConfigKwargs)


