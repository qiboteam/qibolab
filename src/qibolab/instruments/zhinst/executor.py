"""Executing pulse sequences on a Zurich Instruments devices."""

import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Optional

import laboneq.simple as lo
import numpy as np
from qibo.config import log

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.couplers import Coupler

from qibolab.instruments.unrolling import batch_max_sequences
from qibolab.pulses import PulseSequence, PulseType
from qibolab.qubits import Qubit
from qibolab.sweeper import Parameter, Sweeper
from qibolab.unrolling import Bounds

from .abstract import Controller
from .port import Port

from .pulse import ZhPulse
from .sweep import ProcessedSweeps, classify_sweepers
from .util import (
    NANO_TO_SECONDS,
    SAMPLING_RATE,
    acquire_channel_name,
    measure_channel_name,
)

COMPILER_SETTINGS = {
    "SHFSG_MIN_PLAYWAVE_HINT": 32,
    "SHFSG_MIN_PLAYZERO_HINT": 32,
    "HDAWG_MIN_PLAYWAVE_HINT": 64,
    "HDAWG_MIN_PLAYZERO_HINT": 64,
}
"""Translating to Zurich ExecutionParameters."""
ACQUISITION_TYPE = {
    AcquisitionType.INTEGRATION: lo.AcquisitionType.INTEGRATION,
    AcquisitionType.RAW: lo.AcquisitionType.RAW,
    AcquisitionType.DISCRIMINATION: lo.AcquisitionType.DISCRIMINATION,
}

AVERAGING_MODE = {
    AveragingMode.CYCLIC: lo.AveragingMode.CYCLIC,
    AveragingMode.SINGLESHOT: lo.AveragingMode.SINGLE_SHOT,
}


@dataclass
class ZhPort(Port):
    name: tuple[str, str]
    offset: float = 0.0
    power_range: int = 0


@dataclass
class SubSequence:
    """A subsequence is a slice (in time) of a sequence that contains at most
    one measurement per qubit.

    When the driver is asked to execute a sequence, it will first split
    it into sub-sequences. This is needed so that we can create a
    separate laboneq section for each measurement (multiple measurements
    per section are not allowed). When splitting a sequence, it is
    assumed that 1. a measurement operation can be parallel (in time) to
    another measurement operation (i.e. measuring multiple qubits
    simultaneously), but other channels (e.g. drive) do not contain any
    pulses parallel to measurements, 2. ith measurement on some channel
    is in the same subsequence as the ith measurement (if any) on
    another measurement channel, 3. all measurements in one subsequence
    happen at the same time.
    """

    measurements: list[tuple[str, ZhPulse]]
    control_sequence: dict[str, list[ZhPulse]]


class Zurich(Controller):
    """Driver for a collection of ZI instruments that are automatically
    synchronized via ZSync protocol."""

    PortType = ZhPort

    def __init__(self, name, device_setup, time_of_flight=0.0, smearing=0.0):
        super().__init__(name, None)

        self.signal_map = {}
        "Signals to lines mapping"
        self.calibration = lo.Calibration()
        "Zurich calibration object)"

        self.device_setup = device_setup
        self.session = None
        "Zurich device parameters for connection"

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

        self.sequence = defaultdict(list)
        "Zurich pulse sequence"
        self.sub_sequences: list[SubSequence] = []
        "Sub sequences between each measurement"

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
            self.session = lo.Session(self.device_setup, log_level=20)
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
            if len(self.sequence[qubit.drive.name]) != 0:
                self.register_drive_line(
                    qubit=qubit,
                    intermediate_frequency=qubit.drive_frequency
                    - qubit.drive.local_oscillator.frequency,
                )
            if len(self.sequence[measure_channel_name(qubit)]) != 0:
                self.register_readout_line(
                    qubit=qubit,
                    intermediate_frequency=qubit.readout_frequency
                    - qubit.readout.local_oscillator.frequency,
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

        q = qubit.name  # pylint: disable=C0103
        self.signal_map[measure_channel_name(qubit)] = (
            self.device_setup.logical_signal_groups[f"q{q}"].logical_signals[
                "measure_line"
            ]
        )
        self.calibration[f"/logical_signal_groups/q{q}/measure_line"] = (
            lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=intermediate_frequency,
                    modulation_type=lo.ModulationType.SOFTWARE,
                ),
                local_oscillator=lo.Oscillator(
                    frequency=int(qubit.readout.local_oscillator.frequency),
                ),
                range=qubit.readout.power_range,
                port_delay=None,
                delay_signal=0,
            )
        )

        self.signal_map[acquire_channel_name(qubit)] = (
            self.device_setup.logical_signal_groups[f"q{q}"].logical_signals[
                "acquire_line"
            ]
        )

        oscillator = lo.Oscillator(
            frequency=intermediate_frequency,
            modulation_type=lo.ModulationType.SOFTWARE,
        )
        threshold = None

        if options.acquisition_type == AcquisitionType.DISCRIMINATION:
            if qubit.kernel is not None:
                # Kernels don't work with the software modulation on the acquire signal
                oscillator = None
            else:
                # To keep compatibility with angle and threshold discrimination (Remove when possible)
                threshold = qubit.threshold

        self.calibration[f"/logical_signal_groups/q{q}/acquire_line"] = (
            lo.SignalCalibration(
                oscillator=oscillator,
                range=qubit.feedback.power_range,
                port_delay=self.time_of_flight * NANO_TO_SECONDS,
                threshold=threshold,
            )
        )

    def register_drive_line(self, qubit, intermediate_frequency):
        """Registers qubit drive line to calibration and signal map."""
        q = qubit.name  # pylint: disable=C0103
        self.signal_map[qubit.drive.name] = self.device_setup.logical_signal_groups[
            f"q{q}"
        ].logical_signals["drive_line"]
        self.calibration[f"/logical_signal_groups/q{q}/drive_line"] = (
            lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=intermediate_frequency,
                    modulation_type=lo.ModulationType.HARDWARE,
                ),
                local_oscillator=lo.Oscillator(
                    frequency=int(qubit.drive.local_oscillator.frequency),
                ),
                range=qubit.drive.power_range,
                port_delay=None,
                delay_signal=0,
            )
        )

    def register_flux_line(self, qubit):
        """Registers qubit flux line to calibration and signal map."""
        q = qubit.name  # pylint: disable=C0103
        self.signal_map[qubit.flux.name] = self.device_setup.logical_signal_groups[
            f"q{q}"
        ].logical_signals["flux_line"]
        self.calibration[f"/logical_signal_groups/q{q}/flux_line"] = (
            lo.SignalCalibration(
                range=qubit.flux.power_range,
                port_delay=None,
                delay_signal=0,
                voltage_offset=qubit.flux.offset,
            )
        )

    def register_couplerflux_line(self, coupler):
        """Registers qubit flux line to calibration and signal map."""
        c = coupler.name  # pylint: disable=C0103
        self.signal_map[coupler.flux.name] = self.device_setup.logical_signal_groups[
            f"qc{c}"
        ].logical_signals["flux_line"]
        self.calibration[f"/logical_signal_groups/qc{c}/flux_line"] = (
            lo.SignalCalibration(
                range=coupler.flux.power_range,
                port_delay=None,
                delay_signal=0,
                voltage_offset=coupler.flux.offset,
            )
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

    @staticmethod
    def frequency_from_pulses(qubits, sequence):
        """Gets the frequencies from the pulses to the qubits."""
        for pulse in sequence:
            qubit = qubits[pulse.qubit]
            if pulse.type is PulseType.READOUT:
                qubit.readout_frequency = pulse.frequency
            if pulse.type is PulseType.DRIVE:
                qubit.drive_frequency = pulse.frequency

    def create_sub_sequences(self, qubits: list[Qubit]) -> list[SubSequence]:
        """Create subsequences based on locations of measurements."""
        measure_channels = {measure_channel_name(qb) for qb in qubits}
        other_channels = set(self.sequence.keys()) - measure_channels

        measurement_groups = defaultdict(list)
        for ch in measure_channels:
            for i, pulse in enumerate(self.sequence[ch]):
                measurement_groups[i].append((ch, pulse))

        measurement_starts = {}
        for i, group in measurement_groups.items():
            starts = np.array([meas.pulse.start for _, meas in group])
            measurement_starts[i] = max(starts)

        # split all non-measurement channels according to the locations of the measurements
        sub_sequences = defaultdict(lambda: defaultdict(list))
        for ch in other_channels:
            measurement_index = 0
            for pulse in self.sequence[ch]:
                if pulse.pulse.finish > measurement_starts[measurement_index]:
                    measurement_index += 1
                sub_sequences[measurement_index][ch].append(pulse)
        if len(sub_sequences) > len(measurement_groups):
            log.warning("There are control pulses after the last measurement start.")

        return [
            SubSequence(measurement_groups[i], sub_sequences[i])
            for i in range(len(measurement_groups))
        ]

    def experiment_flow(
        self,
        qubits: dict[str, Qubit],
        couplers: dict[str, Coupler],
        sequence: PulseSequence,
        options: ExecutionParameters,
    ):
        """Create the experiment object for the devices, following the steps
        separated one on each method:

        Translation, Calibration, Experiment Definition.

        Args:
            qubits (dict[str, Qubit]): qubits for the platform.
            couplers (dict[str, Coupler]): couplers for the platform.
            sequence (PulseSequence): sequence of pulses to be played in the experiment.
        """
        self.sequence = self.sequence_zh(sequence, qubits)
        self.sub_sequences = self.create_sub_sequences(list(qubits.values()))
        self.calibration_step(qubits, couplers, options)
        self.create_exp(qubits, options)

    # pylint: disable=W0221
    def play(self, qubits, couplers, sequence, options):
        """Play pulse sequence."""
        return self.sweep(qubits, couplers, sequence, options)

    def sequence_zh(
        self, sequence: PulseSequence, qubits: dict[str, Qubit]
    ) -> dict[str, list[ZhPulse]]:
        """Convert Qibo sequence to a sequence where all pulses are replaced
        with ZhPulse instances.

        The resulting object is a dictionary mapping from channel name
        to corresponding sequence of ZhPulse instances
        """
        # Define and assign the sequence
        zhsequence = defaultdict(list)

        # Fill the sequences with pulses according to their lines in temporal order
        for pulse in sequence:
            if pulse.type == PulseType.READOUT:
                ch = measure_channel_name(qubits[pulse.qubit])
            else:
                ch = pulse.channel
            zhsequence[ch].append(ZhPulse(pulse))

        if self.processed_sweeps:
            for ch, zhpulses in zhsequence.items():
                for zhpulse in zhpulses:
                    for param, sweep in self.processed_sweeps.sweeps_for_pulse(
                        zhpulse.pulse
                    ):
                        zhpulse.add_sweeper(param, sweep)

        return zhsequence

    def create_exp(self, qubits, options):
        """Zurich experiment initialization using their Experiment class."""
        if self.acquisition_type:
            acquisition_type = self.acquisition_type
        else:
            acquisition_type = ACQUISITION_TYPE[options.acquisition_type]
        averaging_mode = AVERAGING_MODE[options.averaging_mode]
        exp_options = replace(
            options, acquisition_type=acquisition_type, averaging_mode=averaging_mode
        )

        signals = [lo.ExperimentSignal(name) for name in self.signal_map.keys()]
        exp = lo.Experiment(
            uid="Sequence",
            signals=signals,
        )

        contexts = self._contexts(exp, exp_options)
        self._populate_exp(qubits, exp, exp_options, contexts)
        self.set_calibration_for_rt_sweep(exp)
        exp.set_signal_map(self.signal_map)
        self.experiment = exp

    def _contexts(
        self, exp: lo.Experiment, exp_options: ExecutionParameters
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
        exp: lo.Experiment,
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

    def set_calibration_for_rt_sweep(self, exp: lo.Experiment) -> None:
        """Set laboneq calibration of parameters that are to be swept in real-
        time."""
        if self.processed_sweeps:
            calib = lo.Calibration()
            for ch in (
                set(self.sequence.keys()) | self.processed_sweeps.channels_with_sweeps()
            ):
                for param, sweep_param in self.processed_sweeps.sweeps_for_channel(ch):
                    if param is Parameter.frequency:
                        calib[ch] = lo.SignalCalibration(
                            oscillator=lo.Oscillator(
                                frequency=sweep_param,
                                modulation_type=lo.ModulationType.HARDWARE,
                            )
                        )
            exp.set_calibration(calib)

    def set_instrument_nodes_for_nt_sweep(
        self, exp: lo.Experiment, sweeper: Sweeper
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
        weights = {}
        previous_section = None
        for i, seq in enumerate(self.sub_sequences):
            section_uid = f"control_{i}"
            with exp.section(uid=section_uid, play_after=previous_section):
                for ch, pulses in seq.control_sequence.items():
                    time = 0
                    for pulse in pulses:
                        if pulse.delay_sweeper:
                            exp.delay(signal=ch, time=pulse.delay_sweeper)
                        exp.delay(
                            signal=ch,
                            time=round(pulse.pulse.start * NANO_TO_SECONDS, 9) - time,
                        )
                        time = round(pulse.pulse.duration * NANO_TO_SECONDS, 9) + round(
                            pulse.pulse.start * NANO_TO_SECONDS, 9
                        )
                        if pulse.zhsweepers:
                            self.play_sweep(exp, ch, pulse)
                        else:
                            exp.play(
                                signal=ch,
                                pulse=pulse.zhpulse,
                                phase=pulse.pulse.relative_phase,
                            )
            previous_section = section_uid

            if any(m.delay_sweeper is not None for _, m in seq.measurements):
                section_uid = f"measurement_delay_{i}"
                with exp.section(uid=section_uid, play_after=previous_section):
                    for ch, m in seq.measurements:
                        if m.delay_sweeper:
                            exp.delay(signal=ch, time=m.delay_sweeper)
                previous_section = section_uid

            section_uid = f"measure_{i}"
            with exp.section(uid=section_uid, play_after=previous_section):
                for ch, pulse in seq.measurements:
                    qubit = qubits[pulse.pulse.qubit]
                    q = qubit.name

                    exp.delay(
                        signal=acquire_channel_name(qubit),
                        time=self.smearing * NANO_TO_SECONDS,
                    )

                    if (
                        qubit.kernel is not None
                        and exp_options.acquisition_type
                        == lo.AcquisitionType.DISCRIMINATION
                    ):
                        weight = lo.pulse_library.sampled_pulse_complex(
                            samples=qubit.kernel * np.exp(1j * qubit.iq_angle),
                        )

                    else:
                        if i == 0:
                            if (
                                exp_options.acquisition_type
                                == lo.AcquisitionType.DISCRIMINATION
                            ):
                                weight = lo.pulse_library.sampled_pulse_complex(
                                    samples=np.ones(
                                        [
                                            int(
                                                pulse.pulse.duration * 2
                                                - 3 * self.smearing * NANO_TO_SECONDS
                                            )
                                        ]
                                    )
                                    * np.exp(1j * qubit.iq_angle),
                                )
                                weights[q] = weight
                            else:
                                weight = lo.pulse_library.const(
                                    length=round(
                                        pulse.pulse.duration * NANO_TO_SECONDS, 9
                                    )
                                    - 1.5 * self.smearing * NANO_TO_SECONDS,
                                    amplitude=1,
                                )

                                weights[q] = weight
                        elif i != 0:
                            weight = weights[q]

                    measure_pulse_parameters = {"phase": 0}

                    if i == len(self.sequence[measure_channel_name(qubit)]) - 1:
                        reset_delay = exp_options.relaxation_time * NANO_TO_SECONDS
                    else:
                        reset_delay = 0

                    exp.measure(
                        acquire_signal=acquire_channel_name(qubit),
                        handle=f"sequence{q}_{i}",
                        integration_kernel=weight,
                        integration_kernel_parameters=None,
                        integration_length=None,
                        measure_signal=measure_channel_name(qubit),
                        measure_pulse=pulse.zhpulse,
                        measure_pulse_length=round(
                            pulse.pulse.duration * NANO_TO_SECONDS, 9
                        ),
                        measure_pulse_parameters=measure_pulse_parameters,
                        measure_pulse_amplitude=None,
                        acquire_delay=self.time_of_flight * NANO_TO_SECONDS,
                        reset_delay=reset_delay,
                    )
            previous_section = section_uid

    @staticmethod
    def play_sweep(exp, channel_name, pulse):
        """Play Zurich pulse when a single sweeper is involved."""
        play_parameters = {}
        for p, zhs in pulse.zhsweepers:
            if p is Parameter.amplitude:
                max_value = max(np.abs(zhs.values))
                pulse.zhpulse.amplitude *= max_value
                zhs.values /= max_value
                play_parameters["amplitude"] = zhs
            if p is Parameter.duration:
                play_parameters["length"] = zhs
            if p is Parameter.relative_phase:
                play_parameters["phase"] = zhs
        if "phase" not in play_parameters:
            play_parameters["phase"] = pulse.pulse.relative_phase

        exp.play(signal=channel_name, pulse=pulse.zhpulse, **play_parameters)

    @staticmethod
    def rearrange_rt_sweepers(
        sweepers: list[Sweeper],
    ) -> tuple[Optional[tuple[int, int]], list[Sweeper]]:
        """Rearranges list of real-time sweepers based on hardware limitations.

        The only known limitation currently is that frequency sweepers must be applied before (on the outer loop) other
        (e.g. amplitude) sweepers. Consequently, the only thing done here is to swap the frequency sweeper with the
        first sweeper in the list.

        Args:
            sweepers: Sweepers to rearrange.

        Returns:
            swapped_axis_pair: tuple containing indices of the two swapped axes, or None if nothing to rearrange.
            sweepers: rearranged (or original, if nothing to rearrange) list of sweepers.
        """
        freq_sweeper = next(
            iter(s for s in sweepers if s.parameter is Parameter.frequency), None
        )
        if freq_sweeper:
            sweepers_copy = sweepers.copy()
            freq_sweeper_idx = sweepers_copy.index(freq_sweeper)
            sweepers_copy[freq_sweeper_idx] = sweepers_copy[0]
            sweepers_copy[0] = freq_sweeper
            log.warning("Sweepers were reordered")
            return (0, freq_sweeper_idx), sweepers_copy
        return None, sweepers

    def sweep(self, qubits, couplers, sequence: PulseSequence, options, *sweepers):
        """Play pulse and sweepers sequence."""

        self.signal_map = {}
        self.processed_sweeps = ProcessedSweeps(sweepers, qubits)
        self.nt_sweeps, self.rt_sweeps = classify_sweepers(sweepers)
        swapped_axis_pair, self.rt_sweeps = self.rearrange_rt_sweepers(self.rt_sweeps)
        if swapped_axis_pair:
            # 1. axes corresponding to NT sweeps appear before axes corresponding to RT sweeps
            # 2. in singleshot mode, the first axis contains shots, i.e.: (nshots, sweeper_1, sweeper_2)
            axis_offset = len(self.nt_sweeps) + int(
                options.averaging_mode is AveragingMode.SINGLESHOT
            )
            swapped_axis_pair = tuple(ax + axis_offset for ax in swapped_axis_pair)

        self.frequency_from_pulses(qubits, sequence)

        self.acquisition_type = None
        for sweeper in sweepers:
            if sweeper.parameter in {Parameter.frequency, Parameter.amplitude}:
                for pulse in sweeper.pulses:
                    if pulse.type is PulseType.READOUT:
                        self.acquisition_type = lo.AcquisitionType.SPECTROSCOPY

        self.experiment_flow(qubits, couplers, sequence, options)
        self.run_exp()

        #  Get the results back
        results = {}
        for qubit in qubits.values():
            q = qubit.name  # pylint: disable=C0103
            for i, ropulse in enumerate(self.sequence[measure_channel_name(qubit)]):
                data = self.results.get_data(f"sequence{q}_{i}")

                if swapped_axis_pair:
                    data = np.moveaxis(data, swapped_axis_pair[0], swapped_axis_pair[1])
                if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                    data = (
                        np.ones(data.shape) - data.real
                    )  # Probability inversion patch

                serial = ropulse.pulse.serial
                qubit = ropulse.pulse.qubit
                results[serial] = results[qubit] = options.results_type(data)

        return results

    def split_batches(self, sequences):
        return batch_max_sequences(sequences, MAX_SEQUENCES)
