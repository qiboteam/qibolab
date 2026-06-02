"""Qibolab driver for Keysight QCS instrument set."""

import time
from collections import defaultdict
from functools import reduce
from typing import ClassVar

import numpy as np
from keysight import qcs
from scipy.constants import nano

from qibolab._core.components import AcquisitionChannel, Config, DcChannel, IqChannel
from qibolab._core.execution_parameters import AveragingMode, ExecutionParameters
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import PulseId
from qibolab._core.sequence import InputOps, PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .pulse import (
    process_acquisition_channel_pulse,
    process_dc_channel_pulse,
    process_iq_channel_pulse,
)
from .results import fetch_result, parse_result
from .sweep import RAMP_RATE, process_sweepers

__all__ = ["KeysightQCS"]


def configure_offset(channel: qcs.Channels, offset: float, program: qcs.Program):
    """Configures the DC offset of a given Keysight channel object.

    Arguments:
        channel (qcs.Channels): Keysight channel object.
        offset (float | qcs.Scalar): Channel DC offset.
        program (qcs.Program): Current program being executed.
    """
    program.add_pre_program_operation(
        qcs.SetBaseBandDCOffset(
            channels=channel, constant_voltage=offset, ramping_rate=RAMP_RATE
        )
    )


def sweeper_reducer(
    program: qcs.Program, sweepers: tuple[list[qcs.Array], list[qcs.Scalar]]
):
    """Helper method to unpack the QCS sweep parameters when processing sweepers."""
    return program.sweep(*sweepers)


class KeysightQCS(Controller):
    """Driver for interacting with QCS controller server."""

    qcs_channel_map: qcs.ChannelMapper
    """Map of QCS virtual channels to QCS physical channels."""
    virtual_channel_map: dict[ChannelId, qcs.Channels]
    """Map of Qibolab channel IDs to QCS virtual channels."""
    sampling_rate: ClassVar[float] = (
        qcs.SAMPLE_RATES[qcs.InstrumentEnum.M5300AWG] * nano
    )
    offset_channels: list[ChannelId] = []
    """Subset of channels that require DC offset"""

    def connect(self):
        self.backend = qcs.HclBackend(
            self.qcs_channel_map,
            fpga_postprocessing=True,
            suppress_rounding_warnings=True,
        )
        self.backend.is_system_ready()

    def create_layer(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        sweeper_channel_map: dict[ChannelId, qcs.Scalar],
        sweeper_pulse_map: defaultdict[PulseId, dict[str, qcs.Scalar]],
    ) -> qcs.Layer:

        layer = qcs.Layer()
        # WAVEFORM COMPILATION
        # Iterate over channels and convert qubit pulses to QCS waveforms
        for channel_id, pulse in sequence:
            channel = self.channels[channel_id]
            virtual_channel = self.virtual_channel_map[channel_id]

            if isinstance(channel, AcquisitionChannel):
                probe_channel_id = channel.probe
                classifier_reference = configs[channel_id].state_iq_values
                process_acquisition_channel_pulse(
                    layer=layer,
                    pulse=pulse,
                    frequency=sweeper_channel_map.get(
                        probe_channel_id, configs[probe_channel_id].frequency
                    ),
                    virtual_channel=virtual_channel,
                    probe_virtual_channel=self.virtual_channel_map[probe_channel_id],
                    sweeper_pulse_map=sweeper_pulse_map,
                    classifier=(
                        None
                        if classifier_reference is None
                        else qcs.Classifier(classifier_reference)
                    ),
                )

            elif isinstance(channel, IqChannel):
                process_iq_channel_pulse(
                    layer=layer,
                    pulse=pulse,
                    frequency=sweeper_channel_map.get(
                        channel_id, configs[channel_id].frequency
                    ),
                    virtual_channel=virtual_channel,
                    sweeper_pulse_map=sweeper_pulse_map,
                )

            elif isinstance(channel, DcChannel):
                process_dc_channel_pulse(
                    layer=layer,
                    pulse=pulse,
                    virtual_channel=virtual_channel,
                    sweeper_pulse_map=sweeper_pulse_map,
                )

        return layer

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:
        # Set shot-to-shot delay
        if options.relaxation_time is not None:
            self.backend._init_time = int(options.relaxation_time)
        # Set hardware averaging to speed up result retrival
        averaging = options.averaging_mode is not AveragingMode.SINGLESHOT
        self.backend._publish_every_shot = not averaging

        (
            hardware_sweepers,
            software_sweepers,
            sweeper_channel_map,
            sweeper_pulse_map,
        ) = process_sweepers(sweepers, self.virtual_channel_map)
        # Here we are telling the instrument to run hardware sweepers first, then software sweepers
        # The order of sweeping is determined by the argument order and no re-arrangement is performed
        # @see process_sweepers for hardware/software sweeping decision logic
        program = reduce(
            sweeper_reducer,
            software_sweepers,
            reduce(sweeper_reducer, hardware_sweepers, qcs.Program()).n_shots(
                options.nshots
            ),
        )

        # Configure channel offsets
        for virtual_channel_id in self.offset_channels:
            offset = configs[virtual_channel_id].offset
            virtual_channel = self.virtual_channel_map.get(virtual_channel_id)
            # Do not set DC offset if the channel is being modified by a sweeper
            if virtual_channel not in sweeper_channel_map:
                self.configure_offset(virtual_channel, offset, program)

        acquisition_map: defaultdict[qcs.Channels, list[InputOps]] = defaultdict(list)
        # For each sequence, we assign it to a layer
        # Each layer indicates a sequence of pulses/operations that are synchronized to start at the same time
        # The program will perform all channel operations in a layer before progressing to the next layer

        layers = []
        for sequence in sequences:
            layers.append(
                self.create_layer(
                    sequence.align_to_delays(),
                    configs,
                    sweeper_channel_map,
                    sweeper_pulse_map,
                )
            )
            for channel_id, input_op in sequence.acquisitions:
                channel = self.virtual_channel_map[channel_id]
                acquisition_map[channel].append(input_op)
                # Add time of flight
                time_of_flight = configs[channel_id].delay
                self.qcs_channel_map.get_physical_channels(channel)[
                    0
                ].settings.delay.value = time_of_flight * nano

            # Pad relaxation time delay after each sequence
            if sequence != sequences[-1]:
                layer = qcs.Layer()
                layer.insert(
                    self.virtual_channel_map[next(iter(sequence.channels))],
                    qcs.Delay(duration=options.relaxation_time * nano),
                )
                layers.append(layer)

        program.extend(*layers)
        # FIXME: Instrument experiences random crashes at runtime
        # A partial workaround is to re-launch the job and hope the instrument has not crashed
        try:
            results = self.backend.apply(program).results
        except RuntimeError:
            # If coming from the other branch, rest before attempting a retrial
            time.sleep(0.2)
            results = self.backend.apply(program).results
        # Force a break to prevent race conditions on the instrument side
        time.sleep(0.2)

        ret: dict[PulseId, np.ndarray] = {}
        singleshot_dim = len(software_sweepers) if len(software_sweepers) > 0 else None

        for channel, input_ops in acquisition_map.items():
            raw = fetch_result(
                results=results,
                channel=channel,
                acquisition_type=options.acquisition_type,
                averaging=averaging,
            )

            if len(input_ops) == 1:
                # If only one measurement is requested, raw is at worst an array of
                # (software_sweepers x nshots x hardware_sweepers)
                ret[input_ops[0].id] = parse_result(raw, options, singleshot_dim)
            else:
                # If multiple measurements are requested, raw is instead an array of
                # (software_sweepers x nshots x hardware_sweepers x input_ops)
                for idx, input_op in enumerate(input_ops):
                    ret[input_op.id] = parse_result(
                        raw[..., idx], options, singleshot_dim
                    )

        return ret

    def disconnect(self):
        pass
