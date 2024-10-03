"""Qibolab driver for Keysight QCS instrument set."""

from typing import Optional, Union

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
from qibolab._core.pulses import Drag, Envelope, Gaussian, PulseId, Rectangular
from qibolab._core.sequence import InputOps, PulseSequence
from qibolab._core.sweeper import ParallelSweepers, Parameter
from qibolab._core.unrolling import Bounds

NANOSECONDS = 1e-9

BOUNDS = Bounds(
    waveforms=1,
    readout=1,
    instructions=1,
)

__all__ = ["KeysightQCS"]


def generate_qcs_envelope(shape: Envelope) -> qcs.Envelope:
    """Converts a Qibolab pulse envelope to a QCS Envelope object."""
    if isinstance(shape, Rectangular):
        return qcs.ConstantEnvelope()

    elif isinstance(shape, (Gaussian, Drag)):
        return qcs.GaussianEnvelope(shape.rel_sigma)

    else:
        # raw_envelope = shape.i(num_samples) + 1j * shape.q(num_samples)
        # return qcs.ArbitraryEnvelope(
        #    times=np.linspace(0, 1, num_samples), amplitudes=raw_envelope
        # )
        raise Exception("Envelope not supported")


def generate_qcs_rfwaveform(
    duration: Union[float, qcs.Scalar],
    envelope: Envelope,
    amplitude: Union[float, qcs.Scalar],
    frequency: Union[float, qcs.Scalar],
    phase: Union[float, qcs.Scalar],
) -> qcs.RFWaveform:

    return qcs.RFWaveform(
        duration=duration,
        envelope=generate_qcs_envelope(envelope),
        amplitude=amplitude,
        rf_frequency=frequency,
        instantaneous_phase=phase,
    )


class KeysightQCS(Controller):
    """Driver for interacting with QCS controller server."""

    bounds: str = "qcs/bounds"

    # Map of QCS virtual channels to QCS physical channels
    qcs_channel_map: qcs.ChannelMapper
    # Map of Qibolab channel IDs to QCS virtual channels
    virtual_channel_map: dict[ChannelId, qcs.Channels]
    # Map of QCS virtual acquisition channels to QCS state classifiers
    classifier_map: Optional[dict[qcs.Channels, qcs.MinimumDistanceClassifier]] = None

    def connect(self):
        self.backend = qcs.HclBackend(self.qcs_channel_map, hw_demod=True)
        self.backend.is_system_ready()

    @property
    def sampling_rate(self):
        return qcs.SAMPLE_RATES[qcs.InstrumentEnum.M5300AWG] / 1e9

    def create_program(
        self,
        sequence: PulseSequence,
        configs: dict[str, Config],
        sweepers: list[ParallelSweepers],
        num_shots: int,
    ) -> qcs.Program:
        program = qcs.Program()

        # SWEEPER MANAGEMENT
        # Mapper for pulses that are controlled by a sweeper and the parameter to be swept
        sweeper_pulse_map: dict[PulseId, dict[str, qcs.Scalar]] = {}
        # Mapper for channels with frequency controlled by a sweeper
        sweeper_channel_map: dict[ChannelId, qcs.Scalar] = {}
        hw_demod = True

        for idx, parallel_sweeper in enumerate(sweepers):
            sweep_values = []
            sweep_variables = []

            for idx2, sweeper in enumerate(parallel_sweeper):

                qcs_variable = qcs.Scalar(
                    name=f"V{idx}_{idx2}", value=sweeper.values[0], dtype=float
                )

                if sweeper.parameter is Parameter.frequency:
                    for channel_id in sweeper.channels:
                        sweeper_channel_map[channel_id] = qcs_variable
                        # Ignore hardware sweeping if frequency on readout is swept
                        # TODO: Find better way of determining if channel is readout probe
                        if (
                            "readout" in self.virtual_channel_map[channel_id].name
                            and hw_demod
                        ):
                            program = program.n_shots(num_shots)
                            hw_demod = False
                elif sweeper.parameter in [
                    Parameter.amplitude,
                    Parameter.duration,
                    Parameter.relative_phase,
                ]:
                    # Ignore hardware sweeping if duration is swept
                    if sweeper.parameter is Parameter.duration and hw_demod:
                        program = program.n_shots(num_shots)
                        hw_demod = False
                    for pulse in sweeper.pulses:
                        if pulse.id not in sweeper_pulse_map:
                            sweeper_pulse_map[pulse.id] = {}
                        sweeper_pulse_map[pulse.id][
                            sweeper.parameter.name
                        ] = qcs_variable
                else:
                    raise ValueError(
                        "Sweeper parameter not supported", sweeper.parameter.name
                    )

                sweep_variables.append(qcs_variable)
                sweep_values.append(
                    qcs.Array(
                        name=f"A{idx}_{idx2}",
                        value=(
                            sweeper.values * NANOSECONDS
                            if sweeper.parameter is Parameter.duration
                            else sweeper.values
                        ),
                        dtype=float,
                    )
                )
            program = program.sweep(sweep_values, sweep_variables)

        # WAVEFORM COMPILATION
        # Iterate over channels and convert qubit pulses to QCS waveforms
        for channel_id in sequence.channels:
            channel = self.channels[channel_id]
            virtual_channel = self.virtual_channel_map[channel_id]

            if isinstance(channel, AcquisitionChannel):
                probe_channel_id = channel.probe
                probe_virtual_channel = self.virtual_channel_map[probe_channel_id]
                if probe_channel_id in sweeper_channel_map:
                    frequency = sweeper_channel_map[probe_channel_id]
                else:
                    frequency = configs[probe_channel_id].frequency
                for pulse in sequence.channel(channel_id):
                    sweep_param_map = sweeper_pulse_map.get(pulse.id, {})

                    if pulse.kind == "delay":
                        qcs_pulse = qcs.Delay(
                            sweep_param_map.get(
                                "duration", pulse.duration * NANOSECONDS
                            )
                        )
                        program.add_waveform(qcs_pulse, virtual_channel)
                        program.add_waveform(qcs_pulse, probe_virtual_channel)

                    elif pulse.kind == "acquisition":
                        duration = sweep_param_map.get(
                            "duration", pulse.duration * NANOSECONDS
                        )
                        program.add_acquisition(duration, virtual_channel)

                    elif pulse.kind == "readout":
                        sweep_param_map = sweeper_pulse_map.get(pulse.probe.id, {})
                        qcs_pulse = generate_qcs_rfwaveform(
                            duration=sweep_param_map.get(
                                "duration", pulse.probe.duration * NANOSECONDS
                            ),
                            envelope=pulse.probe.envelope,
                            amplitude=sweep_param_map.get(
                                "amplitude", pulse.probe.amplitude
                            ),
                            frequency=frequency,
                            phase=sweep_param_map.get(
                                "relative_phase", pulse.probe.relative_phase
                            ),
                        )
                        integration_filter = qcs.IntegrationFilter(qcs_pulse)
                        program.add_waveform(qcs_pulse, probe_virtual_channel)
                        program.add_acquisition(integration_filter, virtual_channel)

            elif isinstance(channel, IqChannel):

                pulses = []
                if channel_id in sweeper_channel_map:
                    frequency = sweeper_channel_map[channel_id]
                else:
                    frequency = configs[channel_id].frequency

                vz_phase = 0

                for pulse in sequence.channel(channel_id):
                    sweep_param_map = sweeper_pulse_map.get(pulse.id, {})

                    if pulse.kind == "delay":
                        qcs_pulse = qcs.Delay(
                            sweep_param_map.get(
                                "duration", pulse.duration * NANOSECONDS
                            )
                        )
                    elif pulse.kind == "virtualz":
                        vz_phase += pulse.phase
                    elif pulse.kind == "pulse":
                        qcs_pulse = generate_qcs_rfwaveform(
                            duration=sweep_param_map.get(
                                "duration", pulse.duration * NANOSECONDS
                            ),
                            envelope=pulse.envelope,
                            amplitude=sweep_param_map.get("amplitude", pulse.amplitude),
                            frequency=frequency,
                            phase=sweep_param_map.get(
                                "relative_phase", pulse.relative_phase
                            )
                            + float(vz_phase),
                        )
                        if pulse.envelope.kind == "drag":
                            qcs_pulse = qcs_pulse.drag(coeff=pulse.envelope.beta)
                    else:
                        raise ValueError("Unrecognized pulse type", pulse.kind)

                    pulses.append(qcs_pulse)

                program.add_waveform(pulses, virtual_channel)

            elif isinstance(channel, DcChannel):

                pulses = []
                sweep_param_map = sweeper_pulse_map.get(pulse.id, {})

                for pulse in sequence.channel(channel_id):
                    if pulse.kind == "delay":
                        duration = (
                            sweep_param_map.get(
                                "duration", pulse.duration * NANOSECONDS
                            ),
                        )
                    elif pulse.kind == "pulse":
                        qcs_pulse = qcs.DCWaveform(
                            duration=sweep_param_map.get(
                                "duration", pulse.duration * NANOSECONDS
                            ),
                            envelope=generate_qcs_envelope(pulse.envelope),
                            amplitude=sweep_param_map.get("amplitude", pulse.amplitude),
                        )
                    else:
                        raise ValueError("Unrecognized pulse type", pulse.kind)

                program.add_waveform(pulses, virtual_channel)
        if hw_demod:
            program = program.n_shots(num_shots)
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
            acquisitions = sequence.acquisitions
            acquisition_map: dict[qcs.Channels, list[InputOps]] = {}

            for channel_id, input_op in acquisitions:
                channel = self.virtual_channel_map[channel_id]
                if channel not in acquisition_map:
                    acquisition_map[channel] = []
                acquisition_map[channel].append(input_op)

            averaging = options.averaging_mode is not AveragingMode.SINGLESHOT
            for channel, input_ops in acquisition_map.items():
                if options.acquisition_type is AcquisitionType.RAW:
                    raw = results.get_trace(channel, averaging)
                elif options.acquisition_type is AcquisitionType.INTEGRATION:
                    raw = results.get_iq(channel, averaging)
                elif options.acquisition_type is AcquisitionType.DISCRIMINATION:
                    classifier = self.classifier_map[channel]
                    raw = results.get_classified(channel, averaging, classifier)
                else:
                    raise ValueError("Acquisition type unrecognized")

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
