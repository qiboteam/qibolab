"""Executing pulse sequences on a Zurich Instruments devices."""

import re
from itertools import chain
from typing import Any, Optional, Union

import laboneq.simple as laboneq
import numpy as np
from laboneq.dsl.device import create_connection

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.couplers import Coupler
from qibolab.instruments.abstract import Controller
from qibolab.pulses import Delay, Pulse, PulseSequence
from qibolab.qubits import Qubit
from qibolab.sweeper import Parameter, Sweeper
from qibolab.unrolling import Bounds

from . import ZiChannel
from .pulse import select_pulse
from .sweep import ProcessedSweeps, classify_sweepers
from .util import NANO_TO_SECONDS, SAMPLING_RATE

COMPILER_SETTINGS = {
    "SHFSG_MIN_PLAYWAVE_HINT": 32,
    "SHFSG_MIN_PLAYZERO_HINT": 32,
    "HDAWG_MIN_PLAYWAVE_HINT": 64,
    "HDAWG_MIN_PLAYZERO_HINT": 64,
}
"""Translating to Zurich ExecutionParameters."""
ACQUISITION_TYPE = {
    AcquisitionType.INTEGRATION: laboneq.AcquisitionType.INTEGRATION,
    AcquisitionType.RAW: laboneq.AcquisitionType.RAW,
    AcquisitionType.DISCRIMINATION: laboneq.AcquisitionType.DISCRIMINATION,
}

AVERAGING_MODE = {
    AveragingMode.CYCLIC: laboneq.AveragingMode.CYCLIC,
    AveragingMode.SINGLESHOT: laboneq.AveragingMode.SINGLE_SHOT,
}


class Zurich(Controller):
    """Driver for a collection of ZI instruments that are automatically
    synchronized via ZSync protocol."""

    def __init__(
        self,
        name,
        device_setup,
        channels: list[ZiChannel],
        time_of_flight=0.0,
        smearing=0.0,
    ):
        super().__init__(name, None)

        self.signal_map = {}
        "Signals to lines mapping"
        self.calibration = laboneq.Calibration()
        "Zurich calibration object)"

        for ch in channels:
            device_setup.add_connections(
                ch.device, create_connection(to_signal=ch.name, ports=ch.path)
            )
        self.device_setup = device_setup
        self.session = None
        "Zurich device parameters for connection"

        self.channels = {ch.name: ch for ch in channels}

        self.chanel_to_qubit = {}

        self.time_of_flight = time_of_flight
        self.smearing = smearing
        "Parameters read from the runcard not part of ExecutionParameters"

        self.experiment = None
        self.results = None
        "Zurich experiment definitions"

        self.bounds = Bounds(
            waveforms=int(4e4),
            readout=250,
            instructions=int(1e6),
        )

        self.acquisition_type = None
        "To store if the AcquisitionType.SPECTROSCOPY needs to be enabled by parsing the sequence"

        self.sequences: list[PulseSequence] = []
        "Pulse sequences"

        self.processed_sweeps: Optional[ProcessedSweeps] = None
        self.nt_sweeps: list[Sweeper] = []
        self.rt_sweeps: list[Sweeper] = []

    @property
    def sampling_rate(self):
        return SAMPLING_RATE

    def connect(self):
        if self.is_connected is False:
            # To fully remove logging #configure_logging=False
            # I strongly advise to set it to 20 to have time estimates of the experiment duration!
            self.session = laboneq.Session(self.device_setup, log_level=20)
            _ = self.session.connect()
            self.is_connected = True

    def disconnect(self):
        if self.is_connected:
            _ = self.session.disconnect()
            self.is_connected = False

    def calibration_step(self, qubits, couplers, options):
        """Zurich general pre experiment calibration definitions.

        Change to get frequencies from sequence
        """

        for coupler in couplers.values():
            self.register_couplerflux_line(coupler)

        for qubit in qubits.values():
            if qubit.flux is not None:
                self.register_flux_line(qubit)
            if any(len(seq[qubit.drive.name]) != 0 for seq in self.sequences):
                self.register_drive_line(
                    qubit=qubit,
                    intermediate_frequency=qubit.drive.config.frequency
                    - qubit.drive.config.lo_config.frequency,
                )
            if any(len(seq[qubit.readout.name]) != 0 for seq in self.sequences):
                self.register_readout_line(
                    qubit=qubit,
                    intermediate_frequency=qubit.readout.config.frequency
                    - qubit.readout.config.lo_config.frequency,
                    options=options,
                )
        self.device_setup.set_calibration(self.calibration)

    def register_readout_line(self, qubit, intermediate_frequency, options):
        """Registers qubit measure and acquire lines to calibration and signal
        map.

        Note
        ----
        To allow debugging with and oscilloscope, just set the following::

            self.calibration[f"/logical_signal_groups/q{q}/measure_line"] = lo.SignalCalibration(
                ...,
                local_oscillator=lo.Oscillator(
                    ...
                    frequency=0.0,
                ),
                ...,
                port_mode=lo.PortMode.LF,
                ...,
            )
        """

        measure_signal = self.device_setup.logical_signal_by_uid(qubit.readout.name)
        self.signal_map[qubit.readout.name] = measure_signal
        self.calibration[measure_signal.path] = laboneq.SignalCalibration(
            oscillator=laboneq.Oscillator(
                frequency=intermediate_frequency,
                modulation_type=laboneq.ModulationType.SOFTWARE,
            ),
            local_oscillator=laboneq.Oscillator(
                frequency=int(qubit.readout.config.lo_config.frequency),
            ),
            range=qubit.readout.config.power_range,
            port_delay=None,
            delay_signal=0,
        )

        acquire_signal = self.device_setup.logical_signal_by_uid(qubit.acquisition.name)
        self.signal_map[qubit.acquisition.name] = acquire_signal

        oscillator = laboneq.Oscillator(
            frequency=intermediate_frequency,
            modulation_type=laboneq.ModulationType.SOFTWARE,
        )
        threshold = None

        if options.acquisition_type == AcquisitionType.DISCRIMINATION:
            if qubit.kernel is not None:
                # Kernels don't work with the software modulation on the acquire signal
                oscillator = None
            else:
                # To keep compatibility with angle and threshold discrimination (Remove when possible)
                threshold = qubit.threshold

        self.calibration[acquire_signal.path] = laboneq.SignalCalibration(
            oscillator=oscillator,
            range=qubit.acquisition.config.power_range,
            port_delay=self.time_of_flight * NANO_TO_SECONDS,
            threshold=threshold,
        )

    def register_drive_line(self, qubit, intermediate_frequency):
        """Registers qubit drive line to calibration and signal map."""
        drive_signal = self.device_setup.logical_signal_by_uid(qubit.drive.name)
        self.signal_map[qubit.drive.name] = drive_signal
        self.calibration[drive_signal.path] = laboneq.SignalCalibration(
            oscillator=laboneq.Oscillator(
                frequency=intermediate_frequency,
                modulation_type=laboneq.ModulationType.HARDWARE,
            ),
            local_oscillator=laboneq.Oscillator(
                frequency=int(qubit.drive.config.lo_config.frequency),
            ),
            range=qubit.drive.config.power_range,
            port_delay=None,
            delay_signal=0,
        )

    def register_flux_line(self, qubit):
        """Registers qubit flux line to calibration and signal map."""
        flux_signal = self.device_setup.logical_signal_by_uid(qubit.flux.name)
        self.signal_map[qubit.flux.name] = flux_signal
        self.calibration[flux_signal.name] = laboneq.SignalCalibration(
            range=qubit.flux.config.power_range,
            port_delay=None,
            delay_signal=0,
            voltage_offset=qubit.flux.config.offset,
        )

    def register_couplerflux_line(self, coupler):
        """Registers qubit flux line to calibration and signal map."""
        flux_signal = self.device_setup.logical_signal_by_uid(coupler.flux.name)
        self.signal_map[coupler.flux.name] = flux_signal
        self.calibration[flux_signal.path] = laboneq.SignalCalibration(
            range=coupler.flux.config.power_range,
            port_delay=None,
            delay_signal=0,
            voltage_offset=coupler.flux.config.offset,
        )

    def run_exp(self):
        """
        Compilation settings, compilation step, execution step and data retrival
        - Save a experiment Python object:
        self.experiment.save("saved_exp")
        - Save a experiment compiled experiment ():
        self.exp.save("saved_exp")  # saving compiled experiment
        """
        compiled_experiment = self.session.compile(
            self.experiment, compiler_settings=COMPILER_SETTINGS
        )
        self.results = self.session.run(compiled_experiment)

    def experiment_flow(
        self,
        qubits: dict[str, Qubit],
        couplers: dict[str, Coupler],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
    ):
        """Create the experiment object for the devices, following the steps
        separated one on each method:

        Translation, Calibration, Experiment Definition.

        Args:
            qubits: qubits for the platform.
            couplers: couplers for the platform.
            sequences: list of sequences to be played in the experiment.
            options: execution options/parameters
        """
        self.sequences = sequences
        self.calibration_step(qubits, couplers, options)
        self.create_exp(qubits, options)

    # pylint: disable=W0221
    def play(self, qubits, couplers, sequence, options):
        """Play pulse sequence."""
        return self.sweep(qubits, couplers, sequence, options)

    def create_exp(self, qubits, options):
        """Zurich experiment initialization using their Experiment class."""
        if self.acquisition_type:
            acquisition_type = self.acquisition_type
        else:
            acquisition_type = ACQUISITION_TYPE[options.acquisition_type]
        averaging_mode = AVERAGING_MODE[options.averaging_mode]
        exp_options = options.copy(
            update={
                "acquisition_type": acquisition_type,
                "averaging_mode": averaging_mode,
            }
        )

        signals = [laboneq.ExperimentSignal(name) for name in self.signal_map.keys()]
        exp = laboneq.Experiment(
            uid="Sequence",
            signals=signals,
        )

        contexts = self._contexts(exp, exp_options)
        self._populate_exp(qubits, exp, exp_options, contexts)
        self.set_calibration_for_rt_sweep(exp)
        exp.set_signal_map(self.signal_map)
        self.experiment = exp

    def _contexts(
        self, exp: laboneq.Experiment, exp_options: ExecutionParameters
    ) -> list[tuple[Optional[Sweeper], Any]]:
        """To construct a laboneq experiment, we need to first define a certain
        sequence of nested contexts.

        This method returns the corresponding sequence of context
        managers.
        """
        sweep_contexts = []
        for i, sweeper in enumerate(self.nt_sweeps):
            ctx = exp.sweep(
                uid=f"nt_sweep_{sweeper.parameter.name.lower()}_{i}",
                parameter=[
                    sweep_param
                    for sweep_param in self.processed_sweeps.sweeps_for_sweeper(sweeper)
                ],
            )
            sweep_contexts.append((sweeper, ctx))

        shots_ctx = exp.acquire_loop_rt(
            uid="shots",
            count=exp_options.nshots,
            acquisition_type=exp_options.acquisition_type,
            averaging_mode=exp_options.averaging_mode,
        )
        sweep_contexts.append((None, shots_ctx))

        for i, sweeper in enumerate(self.rt_sweeps):
            ctx = exp.sweep(
                uid=f"rt_sweep_{sweeper.parameter.name.lower()}_{i}",
                parameter=[
                    sweep_param
                    for sweep_param in self.processed_sweeps.sweeps_for_sweeper(sweeper)
                ],
                reset_oscillator_phase=True,
            )
            sweep_contexts.append((sweeper, ctx))

        return sweep_contexts

    def _populate_exp(
        self,
        qubits: dict[str, Qubit],
        exp: laboneq.Experiment,
        exp_options: ExecutionParameters,
        contexts,
    ):
        """Recursively activate the nested contexts, then define the main
        experiment body inside the innermost context."""
        if len(contexts) == 0:
            self.select_exp(exp, qubits, exp_options)
            return

        sweeper, ctx = contexts[0]
        with ctx:
            if sweeper in self.nt_sweeps:
                self.set_instrument_nodes_for_nt_sweep(exp, sweeper)
            self._populate_exp(qubits, exp, exp_options, contexts[1:])

    def set_calibration_for_rt_sweep(self, exp: laboneq.Experiment) -> None:
        """Set laboneq calibration of parameters that are to be swept in real-
        time."""
        if self.processed_sweeps:
            calib = laboneq.Calibration()
            for ch in (
                set(chain(*(seq.keys() for seq in self.sequences)))
                | self.processed_sweeps.channels_with_sweeps()
            ):
                for param, sweep_param in self.processed_sweeps.sweeps_for_channel(ch):
                    if param is Parameter.frequency:
                        calib[ch] = laboneq.SignalCalibration(
                            oscillator=laboneq.Oscillator(
                                frequency=sweep_param,
                                modulation_type=laboneq.ModulationType.HARDWARE,
                            )
                        )
            exp.set_calibration(calib)

    def set_instrument_nodes_for_nt_sweep(
        self, exp: laboneq.Experiment, sweeper: Sweeper
    ) -> None:
        """In some cases there is no straightforward way to sweep a parameter.

        In these cases we achieve sweeping by directly manipulating the
        instrument nodes
        """
        for ch, param, sweep_param in self.processed_sweeps.channel_sweeps_for_sweeper(
            sweeper
        ):
            channel_node_path = self.get_channel_node_path(ch)
            if param is Parameter.bias:
                offset_node_path = f"{channel_node_path}/offset"
                exp.set_node(path=offset_node_path, value=sweep_param)

            # This is supposed to happen only for measurement, but we do not validate it here.
            if param is Parameter.amplitude:
                a, b = re.match(r"(.*)/(\d)/.*", channel_node_path).groups()
                gain_node_path = f"{a}/{b}/oscs/{b}/gain"
                exp.set_node(path=gain_node_path, value=sweep_param)

    def get_channel_node_path(self, channel_name: str) -> str:
        """Return the path of the instrument node corresponding to the given
        channel."""
        logical_signal = self.signal_map[channel_name]
        for instrument in self.device_setup.instruments:
            for conn in instrument.connections:
                if conn.remote_path == logical_signal.path:
                    return f"{instrument.address}/{conn.local_port}"
        raise RuntimeError(
            f"Could not find instrument node corresponding to channel {channel_name}"
        )

    def select_exp(self, exp, qubits, exp_options):
        """Build Zurich Experiment selecting the relevant sections."""
        readout_channels = set(
            chain(*([qb.readout.name, qb.acquisition.name] for qb in qubits.values()))
        )
        weights = {}
        previous_section = None
        for i, seq in enumerate(self.sequences):
            other_channels = set(seq.keys()) - readout_channels
            section_uid = f"sequence_{i}"
            with exp.section(uid=section_uid, play_after=previous_section):
                with exp.section(uid=f"sequence_{i}_control"):
                    for ch in other_channels:
                        for pulse in seq[ch]:
                            self.play_pulse(
                                exp,
                                ch,
                                pulse,
                                self.processed_sweeps.sweeps_for_pulse(pulse),
                            )
                for ch in set(seq.keys()) - other_channels:
                    for j, pulse in enumerate(seq[ch]):
                        with exp.section(uid=f"sequence_{i}_measure_{j}"):
                            qubit = self.chanel_to_qubit[ch]

                            exp.delay(
                                signal=qubit.acquisition.name,
                                time=self.smearing * NANO_TO_SECONDS,
                            )

                            weight = self._get_weights(
                                weights, qubit, pulse, i, exp_options
                            )

                            self.play_pulse(
                                exp,
                                qubit.readout.name,
                                pulse,
                                self.processed_sweeps.sweeps_for_pulse(pulse),
                            )
                            exp.delay(
                                signal=qubit.acquisition.name,
                                time=self.time_of_flight * NANO_TO_SECONDS,
                            )  # FIXME
                            exp.acquire(
                                signal=qubit.acquisition.name,
                                handle=f"sequence{qubit.name}_{i}",
                                kernel=weight,
                            )
                            if j == len(seq[ch]) - 1:
                                exp.delay(
                                    signal=qubit.acquisition.name,
                                    time=exp_options.relaxation_time * NANO_TO_SECONDS,
                                )

            previous_section = section_uid

    def _get_weights(self, weights, qubit, pulse, i, exp_options):
        if (
            qubit.kernel is not None
            and exp_options.acquisition_type == laboneq.AcquisitionType.DISCRIMINATION
        ):
            weight = laboneq.pulse_library.sampled_pulse_complex(
                samples=qubit.kernel * np.exp(1j * qubit.iq_angle),
            )

        else:
            if i == 0:
                if (
                    exp_options.acquisition_type
                    == laboneq.AcquisitionType.DISCRIMINATION
                ):
                    weight = laboneq.pulse_library.sampled_pulse_complex(
                        samples=np.ones(
                            [
                                int(
                                    pulse.duration * 2
                                    - 3 * self.smearing * NANO_TO_SECONDS
                                )
                            ]
                        )
                        * np.exp(1j * qubit.iq_angle),
                    )
                    weights[qubit.name] = weight
                else:
                    weight = laboneq.pulse_library.const(
                        length=round(pulse.duration * NANO_TO_SECONDS, 9)
                        - 1.5 * self.smearing * NANO_TO_SECONDS,
                        amplitude=1,
                    )

                    weights[qubit.name] = weight
            else:
                weight = weights[qubit.name]

        return weight

    @staticmethod
    def play_pulse(
        exp,
        channel: str,
        pulse: Union[Pulse, Delay],
        sweeps: list[tuple[Parameter, laboneq.SweepParameter]],
    ):
        """Play a pulse or delay by taking care of setting parameters to any
        associated sweeps."""
        if isinstance(pulse, Delay):
            duration = pulse.duration
            for p, zhs in sweeps:
                if p is Parameter.duration:
                    duration = zhs
                else:
                    raise ValueError(f"Cannot sweep parameter {p} of a delay.")
            exp.delay(signal=channel, time=duration)
        elif isinstance(pulse, Pulse):
            zhpulse = select_pulse(pulse)
            play_parameters = {}
            for p, zhs in sweeps:
                if p is Parameter.amplitude:
                    max_value = max(np.abs(zhs.values))
                    zhpulse.amplitude *= max_value
                    zhs.values /= max_value
                    play_parameters["amplitude"] = zhs
                if p is Parameter.duration:
                    play_parameters["length"] = zhs
                if p is Parameter.relative_phase:
                    play_parameters["phase"] = zhs
            if "phase" not in play_parameters:
                play_parameters["phase"] = pulse.relative_phase
            exp.play(signal=channel, pulse=zhpulse, **play_parameters)
        else:
            raise ValueError(f"Cannot play pulse: {pulse}")

    def sweep(
        self,
        qubits,
        couplers,
        sequences: list[PulseSequence],
        options,
        *sweepers,
    ):
        """Play pulse and sweepers sequence."""

        self.chanel_to_qubit = {qb.readout.name: qb for qb in qubits.values()}
        self.signal_map = {}
        self.processed_sweeps = ProcessedSweeps(sweepers, self.channels)
        self.nt_sweeps, self.rt_sweeps = classify_sweepers(sweepers)

        self.acquisition_type = None
        readout_channels = set(
            chain(*([qb.readout.name, qb.acquisition.name] for qb in qubits.values()))
        )
        for sweeper in sweepers:
            if sweeper.parameter in {Parameter.frequency}:
                for ch in sweeper.channels:
                    if ch in readout_channels:
                        self.acquisition_type = laboneq.AcquisitionType.SPECTROSCOPY

        self.experiment_flow(qubits, couplers, sequences, options)
        self.run_exp()

        #  Get the results back
        results = {}
        for qubit in qubits.values():
            q = qubit.name  # pylint: disable=C0103
            for i, ropulse in enumerate(
                chain(*(seq[qubit.readout.name] for seq in self.sequences))
            ):
                data = self.results.get_data(f"sequence{q}_{i}")

                if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                    data = (
                        np.ones(data.shape) - data.real
                    )  # Probability inversion patch

                id_ = ropulse.id
                # qubit = ropulse.pulse.qubit
                results[id_] = results[qubit.name] = options.results_type(data)

        return results
