from dataclasses import dataclass, field
from typing import Union

import numpy as np

from qibolab.pulses import Pulse, Rectangular

SAMPLING_RATE = 1
"""Sampling rate of Quantum Machines OPX in GSps."""

__all__ = [
    "operation",
    "Waveform",
    "waveforms_from_pulse",
    "integration_weights",
    "QmPulse",
    "QmAcquisition",
]


def operation(pulse):
    """Generate operation name in QM ``config`` for the given pulse."""
    return str(hash(pulse))


def _wrap(phase: float):
    """Convert phase to multiples of 2pi."""
    return (phase % (2 * np.pi)) / (2 * np.pi)


@dataclass(frozen=True)
class ConstantWaveform:
    sample: float
    type: str = "constant"

    @classmethod
    def from_pulse(cls, pulse: Pulse):
        phase = _wrap(pulse.relative_phase)
        return {
            "I": cls(pulse.amplitude * np.cos(phase)),
            "Q": cls(pulse.amplitude * np.sin(phase)),
        }


@dataclass(frozen=True)
class ArbitraryWaveform:
    samples: list[float]
    type: str = "arbitrary"

    @classmethod
    def from_pulse(cls, pulse: Pulse):
        phase = _wrap(pulse.relative_phase)
        samples_i = pulse.i(SAMPLING_RATE)
        samples_q = pulse.q(SAMPLING_RATE)
        return {
            "I": cls(samples_i * np.cos(phase) - samples_q * np.sin(phase)),
            "Q": cls(samples_i * np.sin(phase) + samples_q * np.cos(phase)),
        }


Waveform = Union[ConstantWaveform, ArbitraryWaveform]


def waveforms_from_pulse(pulse: Pulse) -> Waveform:
    """Register QM waveforms for a given pulse."""
    wvtype = (
        ConstantWaveform
        if isinstance(pulse.envelope, Rectangular)
        else ArbitraryWaveform
    )
    return wvtype.from_pulse(pulse)


@dataclass(frozen=True)
class Waveforms:
    I: str
    Q: str

    @classmethod
    def from_op(cls, op: str):
        return cls(f"{op}_i", f"{op}_q")


@dataclass(frozen=True)
class QmPulse:
    length: int
    waveforms: Union[Waveforms, dict[str, str]]
    digital_marker: str = "ON"
    operation: str = "control"

    @classmethod
    def from_pulse(cls, pulse: Pulse):
        op = operation(pulse)
        return cls(
            length=pulse.duration,
            waveforms=Waveforms.from_op(op),
        )

    @classmethod
    def from_dc_pulse(cls, pulse: Pulse):
        op = operation(pulse)
        return cls(
            length=pulse.duration,
            waveforms={"single": op},
        )


def integration_weights(element: str, readout_len: int, kernel=None, angle: float = 0):
    """Create integration weights section for QM config."""
    cos, sin = np.cos(angle), np.sin(angle)
    if kernel is None:
        convert = lambda x: [(x, readout_len)]
    else:
        cos = kernel * cos
        sin = kernel * sin
        convert = lambda x: x

    return {
        f"cosine_weights_{element}": {
            "cosine": convert(cos),
            "sine": convert(-sin),
        },
        f"sine_weights_{element}": {
            "cosine": convert(sin),
            "sine": convert(cos),
        },
        f"minus_sine_weights_{element}": {
            "cosine": convert(-sin),
            "sine": convert(-cos),
        },
    }


@dataclass(frozen=True)
class QmAcquisition(QmPulse):
    operation: str = "measurement"
    integration_weights: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_pulse(cls, pulse: Pulse, element: str):
        op = operation(pulse)
        integration_weights = {
            "cos": f"cosine_weights_{element}",
            "sin": f"sine_weights_{element}",
            "minus_sin": f"minus_sine_weights_{element}",
        }
        return cls(
            length=pulse.duration,
            waveforms=Waveforms.from_op(op),
            integration_weights=integration_weights,
        )
