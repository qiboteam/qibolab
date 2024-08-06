from dataclasses import dataclass, field
from typing import Union

import numpy as np

__all__ = [
    "output_switch",
    "DcElement",
    "RfOctaveElement",
    "AcquireOctaveElement",
    "Element",
]


def iq_imbalance(g, phi):
    """Creates the correction matrix for the mixer imbalance caused by the gain
    and phase imbalances.

    More information here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer

    Args:
        g (float): relative gain imbalance between the I & Q ports (unit-less).
            Set to 0 for no gain imbalance.
        phi (float): relative phase imbalance between the I & Q ports (radians).
            Set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


@dataclass(frozen=True)
class OutputSwitch:
    port: tuple[str, int]
    delay: int = 57
    buffer: int = 18
    """Default calibration parameters for digital pulses.

    https://docs.quantum-machines.co/1.1.7/qm-qua-sdk/docs/Guides/octave/#calibrating-the-digital-pulse

    Digital markers are used for LO triggering.
    """


def output_switch(opx: str, port: int):
    """Create output switch section."""
    return {"output_switch": OutputSwitch((opx, 2 * port - 1))}


@dataclass
class DcElement:
    singleInput: dict[str, tuple[str, int]]
    intermediate_frequency: int = 0
    operations: dict[str, str] = field(default_factory=dict)


@dataclass
class RfOctaveElement:
    RF_inputs: dict[str, tuple[str, int]]
    digitalInputs: dict[str, OutputSwitch]
    intermediate_frequency: int
    operations: dict[str, str] = field(default_factory=dict)


@dataclass
class AcquireOctaveElement:
    RF_inputs: dict[str, tuple[str, int]]
    RF_outputs: dict[str, tuple[str, int]]
    digitalInputs: dict[str, OutputSwitch]
    intermediate_frequency: int
    time_of_flight: int = 24
    smearing: int = 0
    operations: dict[str, str] = field(default_factory=dict)


Element = Union[DcElement, RfOctaveElement, AcquireOctaveElement]
