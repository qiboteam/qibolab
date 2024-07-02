"""Pre-execution processing of sweeps."""

from collections.abc import Iterable
from copy import copy

import laboneq.simple as laboneq

from qibolab.components import Config
from qibolab.pulses import Pulse, PulseType
from qibolab.sweeper import Parameter, Sweeper

from . import ZiChannel
from .util import NANO_TO_SECONDS


def classify_sweepers(
    sweepers: Iterable[Sweeper],
) -> tuple[list[Sweeper], list[Sweeper]]:
    """Divide sweepers into two lists: 1. sweeps that can be done in the laboneq near-time sweep loop, 2. sweeps that
    can be done in real-time (i.e. on hardware)"""
    nt_sweepers, rt_sweepers = [], []
    for sweeper in sweepers:
        if sweeper.parameter is Parameter.bias or (
            sweeper.parameter is Parameter.amplitude
            and sweeper.pulses[0].type is PulseType.READOUT
        ):
            nt_sweepers.append(sweeper)
        else:
            rt_sweepers.append(sweeper)
    return nt_sweepers, rt_sweepers


class ProcessedSweeps:
    """Data type that centralizes and allows extracting information about given
    sweeps.

    In laboneq, sweeps are represented with the help of SweepParameter
    instances. When adding pulses to a laboneq experiment, some
    properties can be set to be an instance of SweepParameter instead of
    a fixed numeric value. In case of channel property sweeps, either
    the relevant calibration property or the instrument node directly
    can be set ot a SweepParameter instance. Parts of the laboneq
    experiment that define the sweep loops refer to SweepParameter
    instances as well. These should be linkable to instances that are
    either set to a pulse property, a channel calibration or instrument
    node. To achieve this, we use the exact same SweepParameter instance
    in both places. This class takes care of creating these
    SweepParameter instances and giving access to them in a consistent
    way (i.e. whenever they need to be the same instance they will be
    the same instance). When constructing sweep loops you may ask from
    this class to provide all the SweepParameter instances related to a
    given qibolab Sweeper (parallel sweeps). Later, when adding pulses
    or setting channel properties, you may ask from this class to
    provide all SweepParameter instances related to a given pulse or
    channel, and you will get parameters that are linkable to the ones
    in the sweep loop definition
    """

    def __init__(
        self,
        sweepers: Iterable[Sweeper],
        channels: dict[str, ZiChannel],
        configs: dict[str, Config],
    ):
        pulse_sweeps = []
        channel_sweeps = []
        parallel_sweeps = []
        for sweeper in sweepers:
            for pulse in sweeper.pulses or []:
                if sweeper.parameter is Parameter.duration:
                    sweep_param = laboneq.SweepParameter(
                        values=sweeper.values * NANO_TO_SECONDS
                    )
                else:
                    sweep_param = laboneq.SweepParameter(values=copy(sweeper.values))
                pulse_sweeps.append((pulse, sweeper.parameter, sweep_param))
                parallel_sweeps.append((sweeper, sweep_param))

            for ch in sweeper.channels or []:
                logical_channel = channels[ch].logical_channel
                if sweeper.parameter is Parameter.bias:
                    sweep_param = laboneq.SweepParameter(
                        values=sweeper.values + configs[logical_channel.name].offset
                    )
                elif sweeper.parameter is Parameter.frequency:
                    intermediate_frequency = (
                        configs[logical_channel.name].frequency
                        - configs[logical_channel.lo].frequency
                    )
                    sweep_param = laboneq.SweepParameter(
                        values=sweeper.values + intermediate_frequency
                    )
                else:
                    raise ValueError(
                        f"Sweeping {sweeper.parameter.name} for {ch} is not supported"
                    )
                channel_sweeps.append((ch, sweeper.parameter, sweep_param))
                parallel_sweeps.append((sweeper, sweep_param))

        self._pulse_sweeps = pulse_sweeps
        self._channel_sweeps = channel_sweeps
        self._parallel_sweeps = parallel_sweeps

    def sweeps_for_pulse(
        self, pulse: Pulse
    ) -> list[tuple[Parameter, laboneq.SweepParameter]]:
        return [item[1:] for item in self._pulse_sweeps if item[0] == pulse]

    def sweeps_for_channel(
        self, ch: str
    ) -> list[tuple[Parameter, laboneq.SweepParameter]]:
        return [item[1:] for item in self._channel_sweeps if item[0] == ch]

    def sweeps_for_sweeper(self, sweeper: Sweeper) -> list[laboneq.SweepParameter]:
        return [item[1] for item in self._parallel_sweeps if item[0] == sweeper]

    def channel_sweeps_for_sweeper(
        self, sweeper: Sweeper
    ) -> list[tuple[str, Parameter, laboneq.SweepParameter]]:
        return [
            item
            for item in self._channel_sweeps
            if item[2] in self.sweeps_for_sweeper(sweeper)
        ]

    def channels_with_sweeps(self) -> set[str]:
        return {ch for ch, _, _ in self._channel_sweeps}
