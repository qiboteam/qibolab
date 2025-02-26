from dataclasses import dataclass, field
from typing import Union

import numpy as np

from qibolab._core.pulses import Pulse, Rectangular
from qibolab._core.pulses.modulation import rotate, wrap_phase

SAMPLING_RATE = 1
"""Sampling rate of Quantum Machines OPX+ in GSps."""


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
    def from_pulse(cls, pulse: Pulse, max_voltage: float) -> dict[str, "Waveform"]:
        phase = wrap_phase(pulse.relative_phase)
        voltage_amp = pulse.amplitude * max_voltage
        return {
            "I": cls(voltage_amp * np.cos(phase)),
            "Q": cls(voltage_amp * np.sin(phase)),
        }


@dataclass(frozen=True)
class ArbitraryWaveform:
    samples: list[float]
    type: str = "arbitrary"

    @classmethod
    def from_pulse(cls, pulse: Pulse, max_voltage: float) -> dict[str, "Waveform"]:
        original_waveforms = pulse.envelopes(SAMPLING_RATE) * max_voltage
        rotated_waveforms = rotate(original_waveforms, pulse.relative_phase)
        new_duration = baked_duration(pulse.duration)
        pad_len = new_duration - int(pulse.duration)
        baked_waveforms = np.pad(rotated_waveforms, ((0, 0), (0, pad_len)))
        return {
            "I": cls(baked_waveforms[0]),
            "Q": cls(baked_waveforms[1]),
        }


Waveform = Union[ConstantWaveform, ArbitraryWaveform]


def waveforms_from_pulse(pulse: Pulse, max_voltage: float) -> dict[str, Waveform]:
    """Register QM waveforms for a given pulse."""
    needs_baking = pulse.duration < 16 or pulse.duration % 4 != 0
    wvtype = (
        ConstantWaveform
        if isinstance(pulse.envelope, Rectangular) and not needs_baking
        else ArbitraryWaveform
    )
    return wvtype.from_pulse(pulse, max_voltage)


@dataclass(frozen=True)
class Waveforms:
    I: str  # noqa: E741
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


def _convert_integration_weights(x: list[tuple], minus: bool = False) -> list[tuple]:
    """Convert integration weights array for QM."""
    return [(-i[0] if minus else i[0], i[1]) for i in x]


def integration_weights(element: str, readout_len: int, kernel=None, angle: float = 0):
    """Create integration weights section for QM config."""

    if kernel is None:
        cos = [(np.cos(angle), readout_len)]
        sin = [(np.sin(angle), readout_len)]
    else:
        cos = [(i, 4) for i in kernel.real[::4]]
        sin = [(i, 4) for i in kernel.imag[::4]]

    return {
        f"cosine_weights_{element}": {
            "cosine": _convert_integration_weights(cos),
            "sine": _convert_integration_weights(sin, minus=True),
        },
        f"sine_weights_{element}": {
            "cosine": _convert_integration_weights(sin),
            "sine": _convert_integration_weights(cos),
        },
        f"minus_sine_weights_{element}": {
            "cosine": _convert_integration_weights(sin, minus=True),
            "sine": _convert_integration_weights(cos, minus=True),
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
