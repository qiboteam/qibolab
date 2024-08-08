"""Qibolab driver for Keysight QCS instrument set."""

import keysight.qcs as qcs  # pylint: disable=E0401

from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.instruments.abstract import Controller
from qibolab.pulses import (
    Drag,
    Envelope,
    Gaussian,
    PulseSequence,
    PulseType,
    Rectangular,
)
from qibolab.sweeper import ParallelSweepers, Parameter

SWEEPER_PARAMETER_MAP = {Parameter.relative_phase: "instantaneous_phase"}


def generate_envelope_qcs(shape: Envelope):
    """Converts a Qibolab pulse envelope to a QCS Envelope object."""
    if isinstance(shape, Rectangular):
        return qcs.ConstantEnvelope()

    elif isinstance(shape, (Gaussian, Drag)):
        return qcs.GaussianEnvelope(shape.rel_sigma)

    else:
        raise Exception("QCS pulse shape not supported")


class KeysightQCS(Controller):
    """Interaction for interacting with QCS main server."""

    def __init__(
        self,
        name,
        address,
        channel_mapper: qcs.ChannelMapper,
        drive_channel_map: dict[str, qcs.Channels],
        acquire_channel_map: dict[str, qcs.Channels],
    ):
        super().__init__(name, address)
        self.mapper = channel_mapper
        self.drive_channel_map = drive_channel_map
        self.acquire_channel_map = acquire_channel_map

    def connect(self):
        if not self.is_connected:
            self.backend = qcs.HclBackend(
                self.mapper, hw_demod=True, address=self.address
            )
            self.backend.is_system_ready()
            self.is_connected = True

    def play(
        self,
        sequence: PulseSequence,
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ):

        program = qcs.Program()

        # Sweeper management
        scount = 0
        sweeper_pulse_map: dict[int, tuple[str, qcs.Scalar]] = {}
        for parallel_sweeper in sweepers:

            sweeper_arrays = []
            sweeper_variables = []

            for sweeper in parallel_sweeper:
                if sweeper.parameter in [
                    Parameter.attenuation,
                    Parameter.bias,
                    Parameter.gain,
                    Parameter.lo_frequency,
                ]:
                    raise ValueError("Sweeper parameter not supported")

                for pulse in sweeper.pulses:
                    sweeper_name = f"s{scount}"
                    parameter = sweeper.parameter.name
                    qcs_var = qcs.Scalar(sweeper_name, dtype=float)
                    sweeper_variables.append(qcs_var)
                    sweeper_arrays.append(
                        qcs.Array(
                            sweeper_name,
                            value=sweeper.get_values(
                                getattr(pulse, parameter), dtype=float
                            ),
                        )
                    )
                    sweeper_pulse_map[pulse.id] = (
                        SWEEPER_PARAMETER_MAP.get(sweeper.parameter, parameter),
                        qcs_var,
                    )
                    scount += 1

            # For the same sweeper or parallel sweepers, the variables can be swept simultaneously
            program.sweep(sweeper_arrays, sweeper_variables)

        # Map of virtual Z rotations to qubits for phase tracking
        vz_map = {}
        acquisitions = {}

        # Iterate over the pulses in the sequence and add them to the program in order
        for channel, pulses in sequence.items():
            qcs_channel = self.drive_channel_map[channel]
            for pulse in pulses:
                envelope = generate_envelope_qcs(pulse.envelope)

                if pulse.type is PulseType.FLUX or pulse.type is PulseType.COUPLERFLUX:
                    qcs_pulse = qcs.DCWaveform(
                        duration=pulse.duration * 1e-9,
                        envelope=envelope,
                        amplitude=pulse.amplitude,
                    )
                elif pulse.type is PulseType.DELAY:
                    qcs_pulse = qcs.Delay(pulse.duration * 1e-9)
                elif pulse.type is PulseType.VIRTUALZ:
                    # While QCS supports a PhaseIncrement instruction, in our case,
                    # the phase is relative to the qubit and not the channel
                    vz_map[pulse.qubit] = vz_map.get(pulse.qubit, 0) + pulse.phase
                    continue
                else:
                    qcs_pulse = qcs.RFWaveform(
                        duration=pulse.duration * 1e-9,
                        envelope=envelope,
                        amplitude=pulse.amplitude,
                        frequency=pulse.frequency,
                        instantaneous_phase=pulse.relative_phase
                        + vz_map.get(pulse.qubit, 0),
                    )
                    if pulse.type is PulseType.READOUT:
                        readout_channel = self.acquire_channel_map[channel]
                        program.add_acquisition(
                            pulse.duration * 1e-9,
                            readout_channel,
                        )
                        acquisitions[pulse.id] = readout_channel

                    if isinstance(pulse.envelope, Drag):
                        qcs_pulse = qcs_pulse.drag(pulse.shape.beta)

                # If this pulse is part of a sweeper, set the variable
                if pulse.id in sweeper_pulse_map:
                    parameter, qcs_var = sweeper_pulse_map[pulse]
                    setattr(qcs_pulse, parameter, qcs_var)
                program.add_waveform(qcs_pulse, qcs_channel)

        # Set the number of shots
        program.n_shots(options.nshots)
        # Set the relaxation time
        self.backend.init_time = options.relaxation_time * 1e-9
        # Run the program on the backend
        self.backend.apply(program)

        results = {}
        averaging_mode = options.averaging_mode != AveragingMode.SINGLESHOT
        for readout_id, readout_channel in acquisitions.items():
            res = (
                program.get_trace(
                    channels=readout_channel,
                    avg=averaging_mode,
                )
                if options.acquisition_type is AcquisitionType.RAW
                else (
                    program.get_iq(
                        channels=readout_channel,
                        avg=averaging_mode,
                    )
                    if options.acquisition_type is AcquisitionType.INTEGRATION
                    else program.get_classified(
                        channels=readout_channel,
                        avg=averaging_mode,
                    )
                )
            )
            results[readout_id] = res.to_numpy()

        return results
