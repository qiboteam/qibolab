"""Utils for sweeper management."""

from collections import defaultdict

from keysight import qcs
from scipy.constants import nano

from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import PulseId
from qibolab._core.sweeper import ParallelSweepers, Parameter

HARDWARE_SWEEPER_MAX_POINTS = 24576
SUPPORTED_CHANNEL_SWEEPERS = [Parameter.frequency, Parameter.offset]
SUPPORTED_PULSE_SWEEPERS = [
    Parameter.amplitude,
    Parameter.duration,
    Parameter.relative_phase,
    Parameter.phase,
]

RAMP_RATE = 1  # 1V/s
QcsParallelSweep = tuple[list[qcs.Array], list[qcs.Scalar], list[qcs.NonHVIOperation]]


def process_sweepers(
    sweepers: list[ParallelSweepers], virtual_channel_map: dict[ChannelId, qcs.Channels]
):
    """Processes Qibocal sweepers into QCS sweepers. Currently nested hardware
    sweepers are not supported, so they will default to software sweeping.

    Arguments:
        sweepers (list[ParallelSweepers]): Array of array of sweepers.

    Returns:
        hardware_sweepers (list[QcsParallelSweep]): Array of hardware-based QCS sweepers.
        software_sweepers (list[QcsParallelSweep]): Array of software-based QCS sweepers.
        sweeper_channel_map (dict[ChannelId, qcs.Scalar]): Map of channel ID to frequency or offset to be swept.
        sweeper_pulse_map (defaultdict[PulseId, dict[str, qcs.Scalar]]): Map of pulse ID to map of parameter
        to be swept and corresponding QCS variable.
    """
    hardware_sweepers: list[QcsParallelSweep] = []
    software_sweepers: list[QcsParallelSweep] = []
    sweeper_points = 0

    # Mapper for pulses that are controlled by a sweeper and the parameter to be swept
    sweeper_pulse_map: defaultdict[PulseId, dict[str, qcs.Scalar]] = defaultdict(dict)
    # Mapper for channels with frequency controlled by a sweeper
    sweeper_channel_map: dict[ChannelId, qcs.Scalar] = {}

    for idx, parallel_sweeper in enumerate(reversed(sweepers)):
        sweep_values: list[qcs.Array] = []
        sweep_variables: list[qcs.Variable] = []
        # Hardware sweeping is supported up to 8 sweepers
        # If a software sweeper has been declared, every sweeper after must be swept in software
        hardware_sweeping = len(hardware_sweepers) < 9 or len(software_sweepers) == 0
        pre_op_list = []

        for idx2, sweeper in enumerate(parallel_sweeper):
            qcs_variable = qcs.Scalar(
                name=f"V{idx}_{idx2}", value=sweeper.values[0], dtype=float
            )

            if sweeper.parameter in SUPPORTED_CHANNEL_SWEEPERS:
                sweeper_channel_map.update(
                    {channel_id: qcs_variable for channel_id in sweeper.channels}
                )
                # Offset must be software swept
                if sweeper.parameter is Parameter.offset:
                    hardware_sweeping = False
                    virtual_channels = [
                        virtual_channel_map[chan_id] for chan_id in sweeper.channels
                    ]
                    for channel in virtual_channels:
                        pre_op_list.append(
                            qcs.SetBaseBandDCOffset(
                                channel, qcs_variable, ramping_rate=RAMP_RATE
                            )
                        )

            elif sweeper.parameter in SUPPORTED_PULSE_SWEEPERS:
                # Duration can only be swept in hardware for delays
                if sweeper.parameter is Parameter.duration and any(
                    [pulse.kind != "delay" for pulse in sweeper.pulses]
                ):
                    hardware_sweeping = False

                for pulse in sweeper.pulses:
                    sweeper_pulse_map[pulse.id][sweeper.parameter.name] = qcs_variable
            else:
                raise ValueError(
                    "Sweeper parameter not supported", sweeper.parameter.name
                )

            sweep_variables.append(qcs_variable)
            sweep_values.append(
                qcs.Array(
                    name=f"A{idx}_{idx2}",
                    value=(
                        sweeper.values * nano
                        if sweeper.parameter is Parameter.duration
                        else sweeper.values
                    ),
                    dtype=float,
                )
            )
        sweeper_points += len(sweep_values) * len(sweeper.values)
        # For the hardware sweeper, there is a memory limit for the total number of variables x values
        if hardware_sweeping and sweeper_points > HARDWARE_SWEEPER_MAX_POINTS:
            hardware_sweeping = False

        (hardware_sweepers if hardware_sweeping else software_sweepers).append(
            (sweep_values, sweep_variables, pre_op_list)
        )

    return (
        hardware_sweepers,
        software_sweepers,
        sweeper_channel_map,
        sweeper_pulse_map,
    )
