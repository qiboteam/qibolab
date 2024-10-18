"""Qibolab driver for Keysight QCS instrument set."""

from collections import defaultdict
from functools import reduce
from typing import ClassVar, Optional

import keysight.qcs as qcs  # pylint: disable=E0401
import numpy as np

from qibolab._core.components import AcquisitionChannel, Config, DcChannel, IqChannel
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import PulseId
from qibolab._core.sequence import InputOps, PulseSequence
from qibolab._core.sweeper import ParallelSweepers
from qibolab._core.unrolling import Bounds

from .pulse import (
    process_acquisition_channel_pulses,
    process_dc_channel_pulses,
    process_iq_channel_pulses,
)
from .results import fetch_result
from .sweep import process_sweepers

NS_TO_S = 1e-9

BOUNDS = Bounds(
    waveforms=1,
    readout=1,
    instructions=1,
)

__all__ = ["KeysightQCS"]


class KeysightQCS(Controller):
    """Driver for interacting with QCS controller server."""

    bounds: str = "qcs/bounds"

    # Map of QCS virtual channels to QCS physical channels
    qcs_channel_map: qcs.ChannelMapper
    # Map of Qibolab channel IDs to QCS virtual channels
    virtual_channel_map: dict[ChannelId, qcs.Channels]
    # Map of QCS virtual acquisition channels to QCS state classifiers
    classifier_map: Optional[dict[qcs.Channels, qcs.MinimumDistanceClassifier]] = {}
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
            sweeper_swaps_required,
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
                probe_virtual_channel = self.virtual_channel_map[probe_channel_id]
                process_acquisition_channel_pulses(
                    program=program,
                    pulses=sequence.channel(channel_id),
                    frequency=sweeper_channel_map.get(
                        probe_channel_id, configs[probe_channel_id].frequency
                    ),
                    virtual_channel=virtual_channel,
                    probe_virtual_channel=probe_virtual_channel,
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

        return program, sweeper_swaps_required

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
            program, sweeper_swaps_required = self.create_program(
                sequence.align_to_delays(), configs, sweepers, options.nshots
            )
            results = self.backend.apply(program).results
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
                    sweeper_swaps_required=sweeper_swaps_required,
                )

                for result, input_op in zip(raw.values(), input_ops):
                    # For single shot, qibolab expects result format (nshots, ...)
                    # QCS returns (..., nshots), so we need to shuffle the arrays
                    if (
                        options.averaging_mode is AveragingMode.SINGLESHOT
                        and len(sweepers) > 0
                    ):
                        tmp = np.zeros(options.results_shape(sweepers))
                        if options.acquisition_type is AcquisitionType.INTEGRATION:
                            for k in range(options.nshots):
                                tmp[k, ..., 0] = np.real(result[..., k])
                                tmp[k, ..., 1] = np.imag(result[..., k])
                        else:
                            for k in range(options.nshots):
                                tmp[k, ...] = result[..., k]
                        result = tmp

                    elif options.acquisition_type is AcquisitionType.INTEGRATION:
                        tmp = np.zeros(result.shape + (2,))
                        tmp[..., 0] = np.real(result)
                        tmp[..., 1] = np.imag(result)
                        result = tmp
                    ret[input_op.id] = result

        return ret

    def disconnect(self):
        pass
