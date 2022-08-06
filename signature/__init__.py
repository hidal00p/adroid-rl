"""
Module for signature parsing.
Format of Signature Enums are (yamlFieldName, fRequired)
"""

from typing import Union
from signature.conv import SignatureConventions
from signature.extended import NeuralNetConventions, AviaryConventions, ObstacleSensorConventions

ConventionType = Union[
    SignatureConventions,
    NeuralNetConventions,
    AviaryConventions,
    ObstacleSensorConventions
]