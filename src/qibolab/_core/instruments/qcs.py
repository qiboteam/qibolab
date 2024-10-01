"""Qibolab driver for Keysight QCS instrument set."""

from typing import Optional

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
from qibolab._core.pulses import Drag, Envelope, Gaussian, Pulse, PulseId, Rectangular
from qibolab._core.sequence import InputOps, PulseSequence
from qibolab._core.sweeper import ParallelSweepers, Parameter
from qibolab._core.unrolling import Bounds

SWEEPER_PARAMETER_MAP = {Parameter.relative_phase: "instantaneous_phase"}
NANOSECONDS = 1e-9

BOUNDS = Bounds(
    waveforms=1,
    readout=1,
    instructions=1,
)

__all__ = ["KeysightQCS"]


def generate_qcs_envelope(shape: Envelope, num_samples: int) -> qcs.Envelope:
    """Converts a Qibolab pulse envelope to a QCS Envelope object."""
    if isinstance(shape, Rectangular):
        return qcs.ConstantEnvelope()

    elif isinstance(shape, (Gaussian, Drag)):
        return qcs.GaussianEnvelope(shape.rel_sigma)

    else:
        raw_envelope = shape.i(num_samples) + 1j * shape.q(num_samples)
        return qcs.ArbitraryEnvelope(
            times=np.linspace(0, 1, num_samples), amplitudes=raw_envelope
        )


def generate_qcs_rfwaveform(
    pulse: Pulse, sample_rate: float, frequency: float, phase: float = 0
) -> qcs.RFWaveform:
    duration = pulse.duration * NANOSECONDS
    return qcs.RFWaveform(
        duration=duration,
        envelope=generate_qcs_envelope(pulse.envelope, round(duration * sample_rate)),
        amplitude=pulse.amplitude,
        rf_frequency=frequency,
        instantaneous_phase=pulse.relative_phase + phase,
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
    ) -> qcs.Program:
        program = qcs.Program()

        # SWEEPER MANAGEMENT
        # Mapper for pulses that are controlled by a sweeper and the parameter to be swept
        sweeper_pulse_map: dict[PulseId, tuple[qcs.Scalar, Parameter]] = {}
        # Mapper for channels with frequency controlled by a sweeper
        sweeper_channel_map: dict[ChannelId, qcs.Scalar] = {}

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
                elif sweeper.parameter in [
                    Parameter.amplitude,
                    Parameter.frequency,
                    Parameter.duration,
                ]:
                    for pulse in sweeper.pulses:
                        sweeper_pulse_map[pulse.id] = (qcs_variable, sweeper.parameter)
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
                            if Parameter is Parameter.duration
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
            sample_rate = self.qcs_channel_map.get_physical_channels(virtual_channel)[
                0
            ].sample_rate

            if isinstance(channel, AcquisitionChannel):
                probe_channel_id = channel.probe
                probe_virtual_channel = self.virtual_channel_map[probe_channel_id]
                sample_rate = self.qcs_channel_map.get_physical_channels(
                    probe_virtual_channel
                )[0].sample_rate

                if channel_id in sweeper_channel_map:
                    frequency = sweeper_channel_map[probe_channel_id]
                else:
                    frequency = configs[probe_channel_id].frequency

                for pulse in sequence.channel(channel_id):
                    if pulse.kind == "delay":
                        qcs_pulse = qcs.Delay(pulse.duration * NANOSECONDS)
                        if pulse.id in sweeper_pulse_map:
                            qcs_variable, parameter = sweeper_pulse_map[pulse.id]
                            setattr(
                                qcs_pulse,
                                SWEEPER_PARAMETER_MAP.get(parameter, parameter.name),
                                qcs_variable,
                            )
                        program.add_waveform(qcs_pulse, virtual_channel)
                        program.add_waveform(qcs_pulse, probe_virtual_channel)

                    elif pulse.kind == "acquisition":
                        duration = pulse.duration
                        if pulse.id in sweeper_pulse_map:
                            qcs_variable, parameter = sweeper_pulse_map[pulse.id]
                            # Sanity check, but the only parameter for an acquisition operation is the duration
                            if parameter is Parameter.duration:
                                duration = qcs_variable
                        program.add_acquisition(duration, virtual_channel)

                    elif pulse.kind == "readout":
                        qcs_pulse = generate_qcs_rfwaveform(
                            pulse.probe, sample_rate, frequency
                        )
                        if pulse.probe.id in sweeper_pulse_map:
                            qcs_variable, parameter = sweeper_pulse_map[pulse.probe.id]
                            setattr(
                                qcs_pulse,
                                SWEEPER_PARAMETER_MAP.get(parameter, parameter.name),
                                qcs_variable,
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
                    if pulse.kind == "delay":
                        qcs_pulse = qcs.Delay(pulse.duration * NANOSECONDS)
                    elif pulse.kind == "virtualz":
                        vz_phase += pulse.phase
                    elif pulse.kind == "pulse":
                        qcs_pulse = generate_qcs_rfwaveform(
                            pulse, sample_rate, frequency, vz_phase
                        )
                        if pulse.envelope.kind == "drag":
                            qcs_pulse = qcs_pulse.drag(coeff=pulse.envelope.beta)
                    else:
                        raise ValueError("Unrecognized pulse type", pulse.kind)

                    if pulse.id in sweeper_pulse_map:
                        qcs_variable, parameter = sweeper_pulse_map[pulse.id]
                        if parameter is Parameter.relative_phase:
                            qcs_variable += float(vz_phase)
                        setattr(
                            qcs_pulse,
                            SWEEPER_PARAMETER_MAP.get(parameter, parameter.name),
                            qcs_variable,
                        )

                    pulses.append(qcs_pulse)

                program.add_waveform(pulses, virtual_channel)

            elif isinstance(channel, DcChannel):

                pulses = []
                for pulse in sequence.channel(channel_id):
                    if pulse.kind == "delay":
                        qcs_pulse = qcs.Delay(pulse.duration * NANOSECONDS)
                    elif pulse.kind == "pulse":
                        duration = pulse.duration * NANOSECONDS
                        qcs_pulse = qcs.DCWaveform(
                            duration=duration,
                            envelope=generate_qcs_envelope(
                                pulse.envelope, round(sample_rate * duration)
                            ),
                            amplitude=pulse.amplitude,
                        )
                    else:
                        raise ValueError("Unrecognized pulse type", pulse.kind)

                    if pulse.id in sweeper_pulse_map:
                        qcs_variable, parameter = sweeper_pulse_map[pulse.id]
                        setattr(
                            qcs_pulse,
                            SWEEPER_PARAMETER_MAP.get(parameter, parameter.name),
                            qcs_variable,
                        )
                program.add_waveform(pulses, virtual_channel)

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
                    sequence.align_to_delays(), configs, sweepers
                ).n_shots(options.nshots)
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

                for result, input_op in zip(raw.values(), input_ops):
                    if options.acquisition_type is AcquisitionType.INTEGRATION:
                        tmp = np.zeros(result.shape + (2,))
                        tmp[:, 0] = np.real(result)
                        tmp[:, 1] = np.imag(result)
                        result = tmp
                    ret[input_op.id] = result

        return ret

    def disconnect(self):
        pass
