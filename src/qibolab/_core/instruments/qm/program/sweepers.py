import numpy as np
import numpy.typing as npt
from qm import qua
from qm.qua._dsl import _Variable  # for type declaration only

from qibolab._core.components import Channel, Config
from qibolab._core.identifier import ChannelId
from qibolab._core.sweeper import Parameter

from .arguments import ExecutionArguments, Parameters

MAX_AMPLITUDE_FACTOR = 1.99
"""Maximum multiplication factor for ``qua.amp`` used when sweeping amplitude.

https://docs.quantum-machines.co/1.2.0/docs/API_references/qua/dsl_main/#qm.qua._dsl.amp
"""
FREQUENCY_BANDWIDTH = 4e8
"""Quantum Machines OPX+ frequency bandwidth in Hz."""


def find_lo_frequencies(
    args: ExecutionArguments,
    channels: list[tuple[ChannelId, Channel]],
    configs: dict[str, Config],
    values: npt.NDArray,
):
    """Register LO frequencies of swept channels in execution arguments.

    These are needed to calculate the proper IF when sweeping frequency.
    It also checks if frequency sweep is within the supported instrument
    bandwidth [-400, 400] MHz.
    """
    lo_freqs = {configs[channel.lo].frequency for _, channel in channels}
    if len(lo_freqs) > 1:
        raise ValueError(
            "Cannot sweep frequency of channels using different LO using the same `Sweeper` object. Please use parallel sweepers instead."
        )
    lo_frequency = lo_freqs.pop()
    for id, channel in channels:
        max_freq = max(abs(values - lo_frequency))
        if max_freq > FREQUENCY_BANDWIDTH:
            raise ValueError(
                f"Frequency {max_freq} for channel {id} is beyond instrument bandwidth."
            )
        args.parameters[id].lo_frequency = int(lo_frequency)


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


def normalize_frequency(values: npt.NDArray, lo_frequency: int) -> npt.NDArray:
    """Convert frequencies to integer and subtract LO frequency.

    QUA gives an error if the raw frequency values are uploaded to sweep
    over.
    """
    return (values - lo_frequency).astype(int)


def _amplitude(variable: _Variable, parameters: Parameters):
    parameters.amplitude = qua.amp(variable)


def _relative_phase(variable: _Variable, parameters: Parameters):
    parameters.phase = variable


def _duration(variable: _Variable, parameters: Parameters):
    parameters.duration = variable


def _duration_interpolated(variable: _Variable, parameters: Parameters):
    parameters.duration = variable
    parameters.interpolated = True


def _offset(variable: _Variable, parameters: Parameters):
    with qua.if_(variable >= parameters.max_offset):
        qua.set_dc_offset(parameters.element, "single", parameters.max_offset)
    with qua.elif_(variable <= -parameters.max_offset):
        qua.set_dc_offset(parameters.element, "single", -parameters.max_offset)
    with qua.else_():
        qua.set_dc_offset(parameters.element, "single", variable)


def _frequency(variable: _Variable, parameters: Parameters):
    qua.update_frequency(parameters.element, variable)


INT_TYPE = {Parameter.frequency, Parameter.duration, Parameter.duration_interpolated}
"""Sweeper parameters for which we need ``int`` variable type.

The rest parameters need ``fixed`` type.
"""

NORMALIZERS = {
    Parameter.frequency: normalize_frequency,
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
