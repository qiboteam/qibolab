import numpy as np
import numpy.typing as npt
from qibo.config import raise_error
from qm import qua
from qm.qua._dsl import _Variable  # for type declaration only

from qibolab.components import Channel, Config
from qibolab.sweeper import Parameter

from .arguments import Parameters

MAX_OFFSET = 0.5
"""Maximum voltage supported by Quantum Machines OPX+ instrument in volts."""
MAX_AMPLITUDE_FACTOR = 1.99
"""Maximum multiplication factor for ``qua.amp`` used when sweeping amplitude.

https://docs.quantum-machines.co/1.2.0/docs/API_references/qua/dsl_main/#qm.qua._dsl.amp
"""
FREQUENCY_BANDWIDTH = 4e8
"""Quantum Machines OPX+ frequency bandwidth in Hz."""


def check_frequency_bandwidth(
    channels: list[Channel], configs: dict[str, Channel], values: npt.NDArray
):
    """Check if frequency sweep is within the supported instrument bandwidth
    [-400, 400] MHz."""
    for channel in channels:
        name = str(channel.name)
        lo_frequency = configs[channel.lo].frequency
        max_freq = max(abs(values - lo_frequency))
        if max_freq > FREQUENCY_BANDWIDTH:
            raise_error(
                ValueError,
                f"Frequency {max_freq} for channel {name} is beyond instrument bandwidth.",
            )


def sweeper_amplitude(values: npt.NDArray) -> float:
    """Pulse amplitude to be registered in the QM ``config`` when sweeping
    amplitude.

    The multiplicative factor used in the ``qua.amp`` command is limited, so we
    may need to register a pulse with different amplitude than the original pulse
    in the sequence, in order to reach all sweeper values when sweeping amplitude.
    """
    return max(abs(values)) / MAX_AMPLITUDE_FACTOR


def normalize_amplitude(values: npt.NDArray) -> npt.NDArray:
    """Normalize amplitude factor to [-MAX_AMPLITUDE_FACTOR,
    MAX_AMPLITUDE_FACTOR]."""
    return values / sweeper_amplitude(values)


def normalize_phase(values: npt.NDArray) -> npt.NDArray:
    """Normalize phase from [0, 2pi] to [0, 1]."""
    return values / (2 * np.pi)


def normalize_duration(values: npt.NDArray) -> npt.NDArray:
    """Convert duration from ns to clock cycles (clock cycle = 4ns)."""
    if any(values < 16) or not all(values % 4 == 0):
        raise ValueError(
            "Cannot use interpolated duration sweeper for durations that are not multiple of 4ns or are less than 16ns. Please use normal duration sweeper."
        )
    return (values // 4).astype(int)


def _amplitude(variable: _Variable, parameters: Parameters):
    parameters.amplitude = qua.amp(variable)


def _relative_phase(variable: _Variable, parameters: Parameters):
    parameters.phase = variable


def _duration(variable: _Variable, parameters: Parameters):
    parameters.duration = variable


def _duration_interpolated(variable: _Variable, parameters: Parameters):
    parameters.duration = variable
    parameters.interpolated = True


def _offset(variable: _Variable, channel: Channel, configs: dict[str, Config]):
    name = str(channel.name)
    with qua.if_(variable >= MAX_OFFSET):
        qua.set_dc_offset(name, "single", MAX_OFFSET)
    with qua.elif_(variable <= -MAX_OFFSET):
        qua.set_dc_offset(name, "single", -MAX_OFFSET)
    with qua.else_():
        qua.set_dc_offset(name, "single", variable)


def _frequency(variable: _Variable, channel: Channel, configs: dict[str, Config]):
    name = str(channel.name)
    lo_frequency = configs[channel.lo].frequency
    qua.update_frequency(name, variable - lo_frequency)


INT_TYPE = {Parameter.frequency, Parameter.duration, Parameter.duration_interpolated}
"""Sweeper parameters for which we need ``int`` variable type.

The rest parameters need ``fixed`` type.
"""

NORMALIZERS = {
    Parameter.amplitude: normalize_amplitude,
    Parameter.relative_phase: normalize_phase,
    Parameter.duration_interpolated: normalize_duration,
}
"""Functions to normalize sweeper values.

The rest parameters do not need normalization (identity function).
"""

SWEEPER_METHODS = {
    Parameter.frequency: _frequency,
    Parameter.amplitude: _amplitude,
    Parameter.duration: _duration,
    Parameter.duration_interpolated: _duration_interpolated,
    Parameter.relative_phase: _relative_phase,
    Parameter.offset: _offset,
}
"""Methods that return part of QUA program to be used inside the loop."""
