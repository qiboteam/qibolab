"""Utils for sweeper management."""

from collections import defaultdict

from keysight import qcs

from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import PulseId
from qibolab._core.sweeper import ParallelSweepers, Parameter

NS_TO_S = 1e-9
SUPPORTED_CHANNEL_SWEEPERS = [Parameter.frequency]
SUPPORTED_PULSE_SWEEPERS = [
    Parameter.amplitude,
    Parameter.duration,
    Parameter.relative_phase,
]


def process_sweepers(
    sweepers: list[ParallelSweepers], probe_channel_ids: set[ChannelId]
):
    """Processes Qibocal sweepers into QCS sweepers. Currently nested hardware
    sweepers are not supported, so they will default to software sweeping.

    Arguments:
        sweepers (list[ParallelSweepers]): Array of array of sweepers.
        probe_channel_ids (list[int]): Array of channel IDs for probe channels.

    Returns:
        hardware_sweepers (list[tuple[list[qcs.Array], list[qcs.Scalar]]]): Array of hardware-based QCS sweepers.
        software_sweepers (list[tuple[list[qcs.Array], list[qcs.Scalar]]]): Array of software-based QCS sweepers.
        sweeper_swaps_required (list[tuple[int, int]]): Array of corresponding axes to be swapped in results.
        sweeper_channel_map (dict[ChannelId, qcs.Scalar]): Map of channel ID to frequency to be swept.
        sweeper_pulse_map (defaultdict[PulseId, dict[str, qcs.Scalar]]): Map of pulse ID to map of parameter
        to be swept and corresponding QCS variable.
    """
    hardware_sweepers: list[tuple[list[qcs.Array], list[qcs.Scalar]]] = []
    software_sweepers: list[tuple[list[qcs.Array], list[qcs.Scalar]]] = []
    hw_sweeper_order = []
    sw_sweeper_order = []

    # Mapper for pulses that are controlled by a sweeper and the parameter to be swept
    sweeper_pulse_map: defaultdict[PulseId, dict[str, qcs.Scalar]] = defaultdict(dict)
    # Mapper for channels with frequency controlled by a sweeper
    sweeper_channel_map: dict[ChannelId, qcs.Scalar] = {}

    for idx, parallel_sweeper in enumerate(sweepers):
        sweep_values: list[qcs.Array] = []
        sweep_variables: list[qcs.Variable] = []
        # Currently nested hardware sweeping is not supported
        hardware_sweeping = len(hardware_sweepers) == 0

        for idx2, sweeper in enumerate(parallel_sweeper):
            qcs_variable = qcs.Scalar(
                name=f"V{idx}_{idx2}", value=sweeper.values[0], dtype=float
            )

            if sweeper.parameter in SUPPORTED_CHANNEL_SWEEPERS:
                sweeper_channel_map.update(
                    {channel_id: qcs_variable for channel_id in sweeper.channels}
                )
                # Readout frequency is not supported with hardware sweeping
                if not probe_channel_ids.isdisjoint(sweeper.channels):
                    hardware_sweeping = False
            elif sweeper.parameter in SUPPORTED_PULSE_SWEEPERS:
                # Duration is not supported with hardware sweeping
                if sweeper.parameter is Parameter.duration:
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
                        sweeper.values * NS_TO_S
                        if sweeper.parameter is Parameter.duration
                        else sweeper.values
                    ),
                    dtype=float,
                )
            )
        if hardware_sweeping:
            hardware_sweepers.append((sweep_values, sweep_variables))
            hw_sweeper_order.append(idx)
        else:
            software_sweepers.append((sweep_values, sweep_variables))
            sw_sweeper_order.append(idx)

    sweeper_swaps_required = [
        (original_index, shifted_index)
        for shifted_index, original_index in enumerate(
            hw_sweeper_order + sw_sweeper_order
        )
        if original_index != shifted_index
    ]
    return (
        hardware_sweepers,
        software_sweepers,
        sweeper_swaps_required,
        sweeper_channel_map,
        sweeper_pulse_map,
    )
