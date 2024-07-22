from dataclasses import dataclass, field

SAMPLING_RATE = 1
"""Sampling rate of Quantum Machines OPX in GSps."""

__all__ = [
    "operation",
    "Waveform",
    "waveforms_from_pulse",
    "QmPulse",
    "QmAcquisition",
]


def operation(pulse):
    """Generate operation name in QM ``config`` for the given pulse."""
    return str(hash(pulse))


def _normalize_phase(phase: float):
    return (phase % (2 * np.pi)) / (2 * np.pi)


@dataclass(frozen=True)
class ConstantWaveform:
    sample: float
    type: str = "constant"

    @classmethod
    def from_pulse(cls, pulse: Pulse):
        phase = _normalize_phase(pulse.relative_phase)
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
        phase = _normalize_phase(pulse.relative_phase)
        samples_i = pulse.i(SAMPLING_RATE)
        samples_q = pulse.q(SAMPLING_RATE)
        return {
            "I": cls(samples_i * np.cos(phase) - samples_q * np.sin(phase)),
            "Q": cls(samples_i * np.sin(phase) + samples_q * np.cos(phase)),
        }


Waveform = Union[ConstantWaveform, ArbitraryWaveform]


def waveforms_from_pulse(pulse: Pulse) -> Waveform:
    """Register QM waveforms for a given pulse."""
    if isinstance(pulse.envelope, Rectangular):
        return ConstantWaveform.from_pulse(pulse)
    return ArbitraryWaveform.from_pulse(pulse)


@dataclass(frozen=True)
class QmPulse:
    length: int
    waveforms: dict[str, str]
    digital_marker: str = "ON"
    operation: str = "control"

    @classmethod
    def from_pulse(cls, pulse: Pulse):
        op = operation(pulse)
        return cls(
            length=pulse.duration,
            waveforms={"I": f"{op}_i", "Q": f"{op}_q"},
        )

    @classmethod
    def from_dc_pulse(cls, pulse: Pulse):
        op = operation(pulse)
        return cls(
            length=pulse.duration,
            waveforms={"single": op},
        )


def integration_weights(element: str, readout_len: int, kernel=None, angle: float = 0):
    """Registers integration weights in QM config.

    Args:
        readout_len (int): Duration of the readout pulse in ns.
    """
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

    def register_integration_weights(self, element: str, kernel=None, angle: float = 0):
        self.integration_weights = {
            "cos": f"cosine_weights_{element}",
            "sin": f"sine_weights_{element}",
            "minus_sin": f"minus_sine_weights_{element}",
        }
        return integration_weights(element, self.length, kernel, angle)
