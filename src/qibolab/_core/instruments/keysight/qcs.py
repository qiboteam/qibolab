"""Qibolab driver for Keysight QCS instrument set."""

from collections import defaultdict
from functools import reduce
from typing import ClassVar, Optional

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
    process_acquisition_channel_pulses,
    process_dc_channel_pulses,
    process_iq_channel_pulses,
)
from .results import fetch_result, parse_result
from .sweep import process_sweepers

NS_TO_S = 1e-9

__all__ = ["KeysightQCS"]


class KeysightQCS(Controller):
    """Driver for interacting with QCS controller server."""

    bounds: str = "qcs/bounds"

    qcs_channel_map: qcs.ChannelMapper
    """Map of QCS virtual channels to QCS physical channels."""
    virtual_channel_map: dict[ChannelId, qcs.Channels]
    """Map of Qibolab channel IDs to QCS virtual channels."""
    classifier_map: Optional[dict[qcs.Channels, qcs.MinimumDistanceClassifier]] = {}
    """Map of QCS virtual acquisition channels to QCS state classifiers."""
    sampling_rate: ClassVar[float] = (
        qcs.SAMPLE_RATES[qcs.InstrumentEnum.M5300AWG] * NS_TO_S
    )

    def connect(self):
        self.backend = qcs.HclBackend(self.qcs_channel_map, hw_demod=True)
        self.backend.is_system_ready()

    def create_program(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        sweepers: list[ParallelSweepers],
        num_shots: int,
    ) -> tuple[qcs.Program, list[tuple[int, int]]]:
        program = qcs.Program()

        # SWEEPER MANAGEMENT
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
        sweep = lambda program, sweepers: program.sweep(*sweepers)
        program = reduce(
            sweep,
            software_sweepers,
            reduce(sweep, hardware_sweepers, program).n_shots(num_shots),
        )

        # WAVEFORM COMPILATION
        # Iterate over channels and convert qubit pulses to QCS waveforms
        for channel_id in sequence.channels:
            channel = self.channels[channel_id]
            virtual_channel = self.virtual_channel_map[channel_id]

            if isinstance(channel, AcquisitionChannel):
                probe_channel_id = channel.probe
                process_acquisition_channel_pulses(
                    program=program,
                    pulses=sequence.channel(channel_id),
                    frequency=sweeper_channel_map.get(
                        probe_channel_id, configs[probe_channel_id].frequency
                    ),
                    virtual_channel=virtual_channel,
                    probe_virtual_channel=self.virtual_channel_map[probe_channel_id],
                    sweeper_pulse_map=sweeper_pulse_map,
                    classifier=self.classifier_map.get(virtual_channel, None),
                )

            elif isinstance(channel, IqChannel):
                process_iq_channel_pulses(
                    program=program,
                    pulses=sequence.channel(channel_id),
                    frequency=sweeper_channel_map.get(
                        channel_id, configs[channel_id].frequency
                    ),
                    virtual_channel=virtual_channel,
                    sweeper_pulse_map=sweeper_pulse_map,
                )

            elif isinstance(channel, DcChannel):
                process_dc_channel_pulses(
                    program=program,
                    pulses=sequence.channel(channel_id),
                    virtual_channel=virtual_channel,
                    sweeper_pulse_map=sweeper_pulse_map,
                )

        return program

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:

        if options.relaxation_time is not None:
            self.backend._init_time = int(options.relaxation_time)

        ret: dict[PulseId, np.ndarray] = {}
        for sequence in sequences:
            results = self.backend.apply(
                self.create_program(
                    sequence.align_to_delays(), configs, sweepers, options.nshots
                )
            ).results
            acquisition_map: defaultdict[qcs.Channels, list[InputOps]] = defaultdict(
                list
            )

            for channel_id, input_op in sequence.acquisitions:
                channel = self.virtual_channel_map[channel_id]
                acquisition_map[channel].append(input_op)

            averaging = options.averaging_mode is not AveragingMode.SINGLESHOT
            for channel, input_ops in acquisition_map.items():
                raw = fetch_result(
                    results=results,
                    channel=channel,
                    acquisition_type=options.acquisition_type,
                    averaging=averaging,
                )

                for result, input_op in zip(raw.values(), input_ops):

                    ret[input_op.id] = parse_result(result, options)

        return ret

    def disconnect(self):
        pass
