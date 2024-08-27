from dataclasses import dataclass, field
from typing import Union

import numpy as np

from qibolab.pulses import Pulse, Rectangular
from qibolab.pulses.modulation import rotate, wrap_phase

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


def baked_duration(duration: int) -> int:
    """Calculate waveform length after pulse baking.

    QM can only play pulses with length that is >16ns and multiple of
    4ns. Waveforms that don't satisfy these constraints are padded with
    zeros.
    """
    return int(np.maximum((duration + 3) // 4 * 4, 16))


@dataclass(frozen=True)
class ConstantWaveform:
    sample: float
    type: str = "constant"

    @classmethod
    def from_pulse(cls, pulse: Pulse):
        phase = wrap_phase(pulse.relative_phase)
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
        original_waveforms = pulse.envelopes(SAMPLING_RATE)
        rotated_waveforms = rotate(original_waveforms, pulse.relative_phase)
        new_duration = baked_duration(pulse.duration)
        pad_len = new_duration - int(pulse.duration)
        baked_waveforms = np.pad(rotated_waveforms, ((0, 0), (0, pad_len)))
        return {
            "I": cls(baked_waveforms[0]),
            "Q": cls(baked_waveforms[1]),
        }


Waveform = Union[ConstantWaveform, ArbitraryWaveform]


def waveforms_from_pulse(pulse: Pulse) -> Waveform:
    """Register QM waveforms for a given pulse."""
    needs_baking = pulse.duration < 16 or pulse.duration % 4 != 0
    wvtype = (
        ConstantWaveform
        if isinstance(pulse.envelope, Rectangular) and not needs_baking
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
            length=baked_duration(pulse.duration),
            waveforms=Waveforms.from_op(op),
        )

    @classmethod
    def from_dc_pulse(cls, pulse: Pulse):
        op = operation(pulse)
        return cls(
            length=baked_duration(pulse.duration),
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
