"""Qibolab driver for Keysight QCS instrument set."""

import time
from collections import defaultdict
from functools import reduce
from typing import ClassVar

import numpy as np
from keysight import qcs

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
from .sweep import process_sweepers

NS_TO_S = 1e-9

__all__ = ["KeysightQCS"]


def sweeper_reducer(
    program: qcs.Program, sweepers: tuple[list[qcs.Array], list[qcs.Scalar]]
):
    """Helper method to unpack the QCS sweep parameters when processing sweepers."""
    return program.sweep(*sweepers)


class KeysightQCS(Controller):
    """Driver for interacting with QCS controller server."""

    bounds: str = "qcs/bounds"

    qcs_channel_map: qcs.ChannelMapper
    """Map of QCS virtual channels to QCS physical channels."""
    virtual_channel_map: dict[ChannelId, qcs.Channels]
    """Map of Qibolab channel IDs to QCS virtual channels."""
    sampling_rate: ClassVar[float] = (
        qcs.SAMPLE_RATES[qcs.InstrumentEnum.M5300AWG] * NS_TO_S
    )
    offset_channels: list[ChannelId] = []
    """Subset of channels that require offset"""

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
                    pulses=pulse,
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
        if options.relaxation_time is not None:
            self.backend._init_time = int(options.relaxation_time)

        # Configure channel offsets
        workaround_layer = qcs.Layer()
        empty_pulse = qcs.DCWaveform(
            duration=20e-9, amplitude=0, envelope=qcs.ConstantEnvelope()
        )

        for virtual_channel_id in self.offset_channels:
            offset = configs[virtual_channel_id].offset
            virtual_channel = self.virtual_channel_map.get(virtual_channel_id)
            physical_channel = self.backend.channel_mapper.get_physical_channels(
                virtual_channel
            )[0]
            physical_channel.settings.offset.value = offset

            workaround_layer.insert(target=virtual_channel, operations=empty_pulse)

        probe_channel_ids = {
            chan.probe
            for chan in self.channels.values()
            if isinstance(chan, AcquisitionChannel)
        }
        (
            hardware_sweepers,
            software_sweepers,
            sweeper_channel_map,
            sweeper_pulse_map,
        ) = process_sweepers(sweepers, probe_channel_ids)
        # Here we are telling the program to run hardware sweepers first, then software sweepers
        # It is essential that we match the original sweeper order to the modified sweeper order
        # to reconcile the results at the end
        program = reduce(
            sweeper_reducer,
            software_sweepers,
            reduce(sweeper_reducer, hardware_sweepers, qcs.Program()).n_shots(
                options.nshots
            ),
        )

        acquisition_map: defaultdict[qcs.Channels, list[InputOps]] = defaultdict(list)
        # For each sequence, we assign it to a layer
        # Each layer indicates a sequence of pulses/operations that are synchronized to start at the same time
        # The program will perform all channel operations in a layer before progressing to the next layer
        layers = [workaround_layer]
        for sequence in sequences:
            layers.append(
                self.create_layer(
                    sequence, configs, sweeper_channel_map, sweeper_pulse_map
                )
            )
            for channel_id, input_op in sequence.acquisitions:
                channel = self.virtual_channel_map[channel_id]
                acquisition_map[channel].append(input_op)

            # Pad relaxation time delay after each sequence
            if len(sequences) > 1 and sequence != sequences[-1]:
                layer = qcs.Layer()
                layer.insert(
                    next(iter(self.virtual_channel_map.values())),
                    qcs.Delay(duration=options.relaxation_time * NS_TO_S),
                )
                layers.append(layer)

        program.extend(*layers)
        # Retry running the sequence if the program fails at runtime
        try:
            results = self.backend.apply(program).results
            time.sleep(0.2)
        except Exception:
            results = self.backend.apply(program).results

        ret: dict[PulseId, np.ndarray] = {}
        averaging = options.averaging_mode is not AveragingMode.SINGLESHOT
        for channel, input_ops in acquisition_map.items():
            raw = next(
                iter(
                    fetch_result(
                        results=results,
                        channel=channel,
                        acquisition_type=options.acquisition_type,
                        averaging=averaging,
                    ).values()
                )
            )

            if len(input_ops) == 1:
                ret[input_ops[0].id] = parse_result(raw, options)
            else:
                for result, input_op in zip(raw.T, input_ops):
                    ret[input_op.id] = parse_result(result, options)

        return ret

    def disconnect(self):
        pass
