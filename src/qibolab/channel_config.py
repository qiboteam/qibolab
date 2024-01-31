from dataclasses import dataclass
from enum import Enum

"""Definitions for common options for various types of channels. This is part of the end-user API, i.e. is in the top layer.
"""


class AcquisitionType(Enum):
    RAW = "raw"
    INTEGRATION = "integration"
    CLASSIFICATION = "classification"


@dataclass
class DCChannelConfig:
    sampling_rate: float
    bias: float


@dataclass
class IQChannelConfig:
    sampling_rate: float
    frequency: float
    mixer_g: float = 0.0
    mixer_phi: float = 0.0


@dataclass
class OscillatorChannelConfig:
    frequency: float


@dataclass
class AcquisitionChannelConfig:
    type: AcquisitionType
    integration_weights_i: list[float]
    integration_weights_q: list[float]
    classification_kernel: float  # FIXME
