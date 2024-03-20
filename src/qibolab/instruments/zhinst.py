"""Instrument for using the Zurich Instruments (Zhinst) devices."""

import copy
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Union

import laboneq._token
import laboneq.simple as lo
import numpy as np
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation
from laboneq.dsl.experiment.pulse_library import (
    sampled_pulse_complex,
    sampled_pulse_real,
)
from qibo.config import log

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.couplers import Coupler
from qibolab.pulses import CouplerFluxPulse, FluxPulse, PulseSequence, PulseType
from qibolab.qubits import Qubit
from qibolab.sweeper import Parameter, Sweeper
from qibolab.unrolling import Bounds

from .abstract import Controller
from .port import Port

# this env var just needs to be set
os.environ["LABONEQ_TOKEN"] = "not required"
laboneq._token.is_valid_token = lambda _token: True  # pylint: disable=W0212

SAMPLING_RATE = 2
NANO_TO_SECONDS = 1e-9
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

SWEEPER_SET = {"amplitude", "frequency", "duration", "relative_phase"}
SWEEPER_BIAS = {"bias"}
SWEEPER_START = {"start"}


def select_pulse(pulse, pulse_type):
    """Pulse translation."""

    if "IIR" not in str(pulse.shape):
        if str(pulse.shape) == "Rectangular()":
            can_compress = pulse.type is not PulseType.READOUT
            return lo.pulse_library.const(
                uid=(f"{pulse_type}_{pulse.qubit}_"),
                length=round(pulse.duration * NANO_TO_SECONDS, 9),
                amplitude=pulse.amplitude,
                can_compress=can_compress,
            )
        if "Gaussian" in str(pulse.shape):
            sigma = pulse.shape.rel_sigma
            return lo.pulse_library.gaussian(
                uid=(f"{pulse_type}_{pulse.qubit}_"),
                length=round(pulse.duration * NANO_TO_SECONDS, 9),
                amplitude=pulse.amplitude,
                sigma=2 / sigma,
                zero_boundaries=False,
            )

        if "GaussianSquare" in str(pulse.shape):
            sigma = pulse.shape.rel_sigma
            width = pulse.shape.width
            can_compress = pulse.type is not PulseType.READOUT
            return lo.pulse_library.gaussian_square(
                uid=(f"{pulse_type}_{pulse.qubit}_"),
                length=round(pulse.duration * NANO_TO_SECONDS, 9),
                width=round(pulse.duration * NANO_TO_SECONDS, 9) * width,
                amplitude=pulse.amplitude,
                can_compress=can_compress,
                sigma=2 / sigma,
                zero_boundaries=False,
            )

        if "Drag" in str(pulse.shape):
            sigma = pulse.shape.rel_sigma
            beta = pulse.shape.beta
            return lo.pulse_library.drag(
                uid=(f"{pulse_type}_{pulse.qubit}_"),
                length=round(pulse.duration * NANO_TO_SECONDS, 9),
                amplitude=pulse.amplitude,
                sigma=2 / sigma,
                beta=beta,
                zero_boundaries=False,
            )

    if np.all(pulse.envelope_waveform_q(SAMPLING_RATE).data == 0):
        return sampled_pulse_real(
            uid=(f"{pulse_type}_{pulse.qubit}_"),
            samples=pulse.envelope_waveform_i(SAMPLING_RATE).data,
            can_compress=True,
        )
    else:
        # Test this when we have pulses that use it
        return sampled_pulse_complex(
            uid=(f"{pulse_type}_{pulse.qubit}_"),
            samples=pulse.envelope_waveform_i(SAMPLING_RATE).data
            + (1j * pulse.envelope_waveform_q(SAMPLING_RATE).data),
            can_compress=True,
        )

    # Implement Slepian shaped flux pulse https://arxiv.org/pdf/0909.5368.pdf

    # """
    # Typically, the sampler function should discard ``length`` and ``amplitude``, and
    # instead assume that the pulse extends from -1 to 1, and that it has unit
    # amplitude. LabOne Q will automatically rescale the sampler's output to the correct
    # amplitude and length.

    # They don't even do that on their notebooks
    # and just use lenght and amplitude but we have to check


@dataclass
class ZhPort(Port):
    name: Tuple[str, str]
    offset: float = 0.0
    power_range: int = 0


class ZhPulse:
    """Zurich pulse from qibolab pulse translation."""

    def __init__(self, pulse):
        """Zurich pulse from qibolab pulse."""
        self.pulse = pulse
        """Qibolab pulse."""
        self.signal = f"{pulse.type.name.lower()}{pulse.qubit}"
        """Line associated with the pulse."""
        self.zhpulse = select_pulse(pulse, pulse.type.name.lower())
        """Zurich pulse."""


class ZhSweeper:
    """Zurich sweeper from qibolab sweeper for pulse parameters Amplitude,
    Duration, Frequency (and maybe Phase)"""

    def __init__(self, pulse, sweeper, qubit):
        self.sweeper = sweeper
        """Qibolab sweeper."""

        self.pulse = pulse
        """Qibolab pulse associated to the sweeper."""
        self.signal = f"{pulse.type.name.lower()}{pulse.qubit}"
        """Line associated with the pulse."""
        self.zhpulse = ZhPulse(pulse).zhpulse
        """Zurich pulse associated to the sweeper."""

        self.zhsweeper = self.select_sweeper(pulse.type, sweeper, qubit)
        """Zurich sweeper."""

        self.zhsweepers = [self.select_sweeper(pulse.type, sweeper, qubit)]
        """Zurich sweepers, Need something better to store multiple sweeps on
        the same pulse.

        Not properly implemented as it was only used on Rabi amplitude
        vs lenght and it was an unused routine.
        """

    @staticmethod  # pylint: disable=R0903
    def select_sweeper(ptype, sweeper, qubit):
        """Sweeper translation."""

        if sweeper.parameter is Parameter.amplitude:
            return lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=copy.copy(sweeper.values),
            )
        if sweeper.parameter is Parameter.duration:
            return lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values * NANO_TO_SECONDS,
            )
        if sweeper.parameter is Parameter.relative_phase:
            return lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values,
            )
        if sweeper.parameter is Parameter.frequency:
            if ptype is PulseType.READOUT:
                intermediate_frequency = (
                    qubit.readout_frequency - qubit.readout.local_oscillator.frequency
                )
            elif ptype is PulseType.DRIVE:
                intermediate_frequency = (
                    qubit.drive_frequency - qubit.drive.local_oscillator.frequency
                )
            return lo.LinearSweepParameter(
                uid=sweeper.parameter.name,
                start=sweeper.values[0] + intermediate_frequency,
                stop=sweeper.values[-1] + intermediate_frequency,
                count=len(sweeper.values),
            )

    def add_sweeper(self, sweeper, qubit):
        """Add sweeper to list of sweepers."""
        self.zhsweepers.append(self.select_sweeper(self.pulse.type, sweeper, qubit))


class ZhSweeperLine:
    """Zurich sweeper from qibolab sweeper for non pulse parameters Bias, Delay
    (, power_range, local_oscillator frequency, offset ???)

    For now Parameter.bias sweepers are implemented as
    Parameter.Amplitude on a flux pulse. We may want to keep this class
    separate for future Near Time sweeps
    """

    def __init__(self, sweeper, qubit=None, sequence=None, pulse=None):
        self.sweeper = sweeper
        """Qibolab sweeper."""

        # Do something with the pulse coming here
        if sweeper.parameter is Parameter.bias:
            if isinstance(qubit, Qubit):
                pulse = FluxPulse(
                    start=0,
                    duration=sequence.duration + sequence.start,
                    amplitude=1,
                    shape="Rectangular",
                    channel=qubit.flux.name,
                    qubit=qubit.name,
                )
                self.signal = f"flux{qubit.name}"
            if isinstance(qubit, Coupler):
                pulse = CouplerFluxPulse(
                    start=0,
                    duration=sequence.duration + sequence.start,
                    amplitude=1,
                    shape="Rectangular",
                    channel=qubit.flux.name,
                    qubit=qubit.name,
                )
                self.signal = f"couplerflux{qubit.name}"

            self.pulse = pulse

            self.zhpulse = lo.pulse_library.const(
                uid=(f"{pulse.type.name.lower()}_{pulse.qubit}_"),
                length=round(pulse.duration * NANO_TO_SECONDS, 9),
                amplitude=pulse.amplitude,
            )

        elif sweeper.parameter is Parameter.start:
            if pulse:
                self.pulse = pulse
                self.signal = f"flux{qubit}"

                self.zhpulse = ZhPulse(pulse).zhpulse

        # Need something better to store multiple sweeps on the same pulse
        self.zhsweeper = self.select_sweeper(sweeper)

    @staticmethod  # pylint: disable=R0903
    def select_sweeper(sweeper):
        """Sweeper translation."""
        if sweeper.parameter is Parameter.bias:
            return lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values,
            )
        if sweeper.parameter is Parameter.start:
            return lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values * NANO_TO_SECONDS,
            )


class Zurich(Controller):
    """Zurich driver main class."""

    PortType = ZhPort

    def __init__(
        self, name, device_setup, use_emulation=False, time_of_flight=0.0, smearing=0.0
    ):
        self.name = name
        "Setup name (str)"

        self.emulation = use_emulation
        "Enable emulation mode (bool)"
        self.is_connected = False
        "Is the device connected ? (bool)"

        self.signal_map = {}
        "Signals to lines mapping"
        self.calibration = lo.Calibration()
        "Zurich calibration object)"

        self.device_setup = device_setup
        self.session = None
        self.device = None
        "Zurich device parameters for connection"

        self.time_of_flight = time_of_flight
        self.smearing = smearing
        self.chip = "iqm5q"
        "Parameters read from the runcard not part of ExecutionParameters"

        self.exp = None
        self.experiment = None
        self.exp_options = ExecutionParameters()
        self.exp_calib = lo.Calibration()
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
        self.sequence_qibo = None
        # Remove if able
        self.sub_sequences = {}
        "Sub sequences between each measurement"

        self.sweepers = []
        self.nt_sweeps = None
        "Storing sweepers"
        # Improve the storing of multiple sweeps
        self._ports = {}
        self.settings = None

    @property
    def sampling_rate(self):
        return SAMPLING_RATE

    def connect(self):
        if self.is_connected is False:
            # To fully remove logging #configure_logging=False
            # I strongly advise to set it to 20 to have time estimates of the experiment duration!
            self.session = lo.Session(self.device_setup, log_level=20)
            self.device = self.session.connect(do_emulation=self.emulation)
            self.is_connected = True

    def disconnect(self):
        if self.is_connected:
            self.device = self.session.disconnect()
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
            if len(self.sequence[f"drive{qubit.name}"]) != 0:
                self.register_drive_line(
                    qubit=qubit,
                    intermediate_frequency=qubit.drive_frequency
                    - qubit.drive.local_oscillator.frequency,
                )
            if len(self.sequence[f"readout{qubit.name}"]) != 0:
                self.register_readout_line(
                    qubit=qubit,
                    intermediate_frequency=qubit.readout_frequency
                    - qubit.readout.local_oscillator.frequency,
                    options=options,
                )
                if options.fast_reset is not False:
                    if len(self.sequence[f"drive{qubit.name}"]) == 0:
                        self.register_drive_line(
                            qubit=qubit,
                            intermediate_frequency=qubit.drive_frequency
                            - qubit.drive.local_oscillator.frequency,
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
        self.signal_map[f"measure{q}"] = self.device_setup.logical_signal_groups[
            f"q{q}"
        ].logical_signals["measure_line"]
        self.calibration[f"/logical_signal_groups/q{q}/measure_line"] = (
            lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=intermediate_frequency,
                    modulation_type=lo.ModulationType.SOFTWARE,
                ),
                local_oscillator=lo.Oscillator(
                    uid="lo_shfqa_m" + str(q),
                    frequency=int(qubit.readout.local_oscillator.frequency),
                ),
                range=qubit.readout.power_range,
                port_delay=None,
                delay_signal=0,
            )
        )

        self.signal_map[f"acquire{q}"] = self.device_setup.logical_signal_groups[
            f"q{q}"
        ].logical_signals["acquire_line"]

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
        self.signal_map[f"drive{q}"] = self.device_setup.logical_signal_groups[
            f"q{q}"
        ].logical_signals["drive_line"]
        self.calibration[f"/logical_signal_groups/q{q}/drive_line"] = (
            lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=intermediate_frequency,
                    modulation_type=lo.ModulationType.HARDWARE,
                ),
                local_oscillator=lo.Oscillator(
                    uid="lo_shfqc" + str(q),
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
        self.signal_map[f"flux{q}"] = self.device_setup.logical_signal_groups[
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
        self.signal_map[f"couplerflux{c}"] = self.device_setup.logical_signal_groups[
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
        self.exp = self.session.compile(
            self.experiment, compiler_settings=COMPILER_SETTINGS
        )
        # self.exp.save_compiled_experiment("saved_exp")
        self.results = self.session.run(self.exp)

    @staticmethod
    def frequency_from_pulses(qubits, sequence):
        """Gets the frequencies from the pulses to the qubits."""
        # Implement Dual drive frequency experiments, we don't have any for now
        for pulse in sequence:
            qubit = qubits[pulse.qubit]
            if pulse.type is PulseType.READOUT:
                qubit.readout_frequency = pulse.frequency
            if pulse.type is PulseType.DRIVE:
                qubit.drive_frequency = pulse.frequency

    def create_sub_sequence(
        self,
        line_name: str,
        quantum_elements: Union[Dict[str, Qubit], Dict[str, Coupler]],
    ):
        """Create a list of sequences for each measurement.

        Args:
            line_name (str): Name of the line from which extract the sequence.
            quantum_elements (dict[str, Qubit]|dict[str, Coupler]): qubits or couplers for the platform.
        """
        for quantum_element in quantum_elements.values():
            q = quantum_element.name  # pylint: disable=C0103
            measurements = self.sequence[f"readout{q}"]
            pulses = self.sequence[f"{line_name}{q}"]
            pulse_sequences = [[] for _ in measurements]
            pulse_sequences.append([])
            measurement_index = 0
            for pulse in pulses:
                if measurement_index < len(measurements):
                    if pulse.pulse.finish > measurements[measurement_index].pulse.start:
                        measurement_index += 1
                pulse_sequences[measurement_index].append(pulse)
            self.sub_sequences[f"{line_name}{q}"] = pulse_sequences

    def create_sub_sequences(
        self, qubits: Dict[str, Qubit], couplers: Dict[str, Coupler]
    ):
        """Create subsequences for different lines (drive, flux, coupler flux).

        Args:
            qubits (dict[str, Qubit]): qubits for the platform.
            couplers (dict[str, Coupler]): couplers for the platform.
        """
        self.sub_sequences = {}
        self.create_sub_sequence("drive", qubits)
        self.create_sub_sequence("flux", qubits)
        self.create_sub_sequence("couplerflux", couplers)

    def experiment_flow(
        self,
        qubits: Dict[str, Qubit],
        couplers: Dict[str, Coupler],
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
        self.sequence_zh(sequence, qubits, couplers)
        self.create_sub_sequences(qubits, couplers)
        self.calibration_step(qubits, couplers, options)
        self.create_exp(qubits, couplers, options)

    # pylint: disable=W0221
    def play(self, qubits, couplers, sequence, options):
        """Play pulse sequence."""
        self.signal_map = {}

        self.frequency_from_pulses(qubits, sequence)

        self.experiment_flow(qubits, couplers, sequence, options)

        self.run_exp()

        # Get the results back
        results = {}
        for qubit in qubits.values():
            q = qubit.name  # pylint: disable=C0103
            if len(self.sequence[f"readout{q}"]) != 0:
                for i, ropulse in enumerate(self.sequence[f"readout{q}"]):
                    data = np.array(self.results.get_data(f"sequence{q}_{i}"))
                    if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                        data = (
                            np.ones(data.shape) - data.real
                        )  # Probability inversion patch
                    serial = ropulse.pulse.serial
                    qubit = ropulse.pulse.qubit
                    results[serial] = results[qubit] = options.results_type(data)

        # html containing the pulse sequence schedule
        # lo.show_pulse_sheet("pulses", self.exp)
        return results

    def sequence_zh(self, sequence, qubits, couplers):
        """Qibo sequence to Zurich sequence."""
        # Define and assign the sequence
        zhsequence = defaultdict(list)
        self.sequence_qibo = sequence

        # Fill the sequences with pulses according to their lines in temporal order
        for pulse in sequence:
            zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"].append(ZhPulse(pulse))

        # Mess that gets the sweeper and substitutes the pulse it sweeps in the right place

        def nt_loop(sweeper):
            if not self.nt_sweeps:
                self.nt_sweeps = [sweeper]
            else:
                self.nt_sweeps.append(sweeper)
            self.sweepers.remove(sweeper)

        for sweeper in self.sweepers.copy():
            if sweeper.parameter.name in SWEEPER_SET:
                for pulse in sweeper.pulses:
                    aux_list = zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"]
                    if (
                        sweeper.parameter is Parameter.frequency
                        and pulse.type is PulseType.READOUT
                    ):
                        self.acquisition_type = lo.AcquisitionType.SPECTROSCOPY
                    if (
                        sweeper.parameter is Parameter.amplitude
                        and pulse.type is PulseType.READOUT
                    ):
                        self.acquisition_type = lo.AcquisitionType.SPECTROSCOPY
                        nt_loop(sweeper)
                    for element in aux_list:
                        if pulse == element.pulse:
                            if isinstance(aux_list[aux_list.index(element)], ZhPulse):
                                if isinstance(pulse, CouplerFluxPulse):
                                    aux_list[aux_list.index(element)] = ZhSweeper(
                                        pulse, sweeper, couplers[pulse.qubit]
                                    )
                                else:
                                    aux_list[aux_list.index(element)] = ZhSweeper(
                                        pulse, sweeper, qubits[pulse.qubit]
                                    )
                            elif isinstance(
                                aux_list[aux_list.index(element)], ZhSweeper
                            ):
                                if isinstance(pulse, CouplerFluxPulse):
                                    aux_list[aux_list.index(element)].add_sweeper(
                                        sweeper, couplers[pulse.qubit]
                                    )
                                else:
                                    aux_list[aux_list.index(element)].add_sweeper(
                                        sweeper, qubits[pulse.qubit]
                                    )

            if sweeper.parameter.name in SWEEPER_BIAS:
                nt_loop(sweeper)

            # This may not place the Zhsweeper when the start occurs among different sections or lines
            if sweeper.parameter.name in SWEEPER_START:
                pulse = sweeper.pulses[0]
                aux_list = zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"]
                for element in aux_list:
                    if pulse == element.pulse:
                        if isinstance(aux_list[aux_list.index(element)], ZhPulse):
                            aux_list.insert(
                                aux_list.index(element),
                                ZhSweeperLine(sweeper, pulse.qubit, sequence, pulse),
                            )
                            break

        self.sequence = zhsequence

    def create_exp(self, qubits, couplers, options):
        """Zurich experiment initialization using their Experiment class."""

        # Setting experiment signal lines
        signals = []
        for coupler in couplers.values():
            signals.append(lo.ExperimentSignal(f"couplerflux{coupler.name}"))

        for qubit in qubits.values():
            q = qubit.name  # pylint: disable=C0103
            if len(self.sequence[f"drive{q}"]) != 0:
                signals.append(lo.ExperimentSignal(f"drive{q}"))
            if qubit.flux is not None:
                signals.append(lo.ExperimentSignal(f"flux{q}"))
            if len(self.sequence[f"readout{q}"]) != 0:
                signals.append(lo.ExperimentSignal(f"measure{q}"))
                signals.append(lo.ExperimentSignal(f"acquire{q}"))
                if options.fast_reset is not False:
                    if len(self.sequence[f"drive{q}"]) == 0:
                        signals.append(lo.ExperimentSignal(f"drive{q}"))

        exp = lo.Experiment(
            uid="Sequence",
            signals=signals,
        )

        if self.acquisition_type:
            acquisition_type = self.acquisition_type
            self.acquisition_type = None
        else:
            acquisition_type = ACQUISITION_TYPE[options.acquisition_type]
        averaging_mode = AVERAGING_MODE[options.averaging_mode]
        exp_options = replace(
            options, acquisition_type=acquisition_type, averaging_mode=averaging_mode
        )

        exp_calib = lo.Calibration()
        # Near Time recursion loop or directly to Real Time recursion loop
        if self.nt_sweeps is not None:
            self.sweep_recursion_nt(qubits, couplers, exp_options, exp, exp_calib)
        else:
            self.define_exp(qubits, couplers, exp_options, exp, exp_calib)

    def define_exp(self, qubits, couplers, exp_options, exp, exp_calib):
        """Real time definition."""
        with exp.acquire_loop_rt(
            uid="shots",
            count=exp_options.nshots,
            acquisition_type=exp_options.acquisition_type,
            averaging_mode=exp_options.averaging_mode,
        ):
            # Recursion loop for sweepers or just play a sequence
            if len(self.sweepers) > 0:
                self.sweep_recursion(qubits, couplers, exp, exp_calib, exp_options)
            else:
                self.select_exp(exp, qubits, couplers, exp_options)
            exp.set_calibration(exp_calib)
            exp.set_signal_map(self.signal_map)
            self.experiment = exp

    def select_exp(self, exp, qubits, couplers, exp_options):
        """Build Zurich Experiment selecting the relevant sections."""
        if "coupler" in str(self.sequence):
            self.couplerflux(exp, couplers)
        if "drive" in str(self.sequence):
            if "flux" in str(self.sequence):
                self.flux(exp, qubits)
                self.drive(exp, qubits)
            else:
                self.drive(exp, qubits)
        elif "flux" in str(self.sequence):
            self.flux(exp, qubits)
        self.measure_relax(
            exp,
            qubits,
            couplers,
            exp_options.relaxation_time,
            exp_options.acquisition_type,
        )
        if exp_options.fast_reset is not False:
            self.fast_reset(exp, qubits, exp_options.fast_reset)

    @staticmethod
    def play_sweep_select_single(exp, qubit, pulse, section, parameters, partial_sweep):
        """Play Zurich pulse when a single sweeper is involved."""
        if any("amplitude" in param for param in parameters):
            pulse.zhpulse.amplitude *= max(pulse.zhsweeper.values)
            pulse.zhsweeper.values /= max(pulse.zhsweeper.values)
            exp.play(
                signal=f"{section}{qubit.name}",
                pulse=pulse.zhpulse,
                amplitude=pulse.zhsweeper,
                phase=pulse.pulse.relative_phase,
            )
        elif any("duration" in param for param in parameters):
            exp.play(
                signal=f"{section}{qubit.name}",
                pulse=pulse.zhpulse,
                length=pulse.zhsweeper,
                phase=pulse.pulse.relative_phase,
            )
        elif any("relative_phase" in param for param in parameters):
            exp.play(
                signal=f"{section}{qubit.name}",
                pulse=pulse.zhpulse,
                phase=pulse.zhsweeper,  # I believe this is the global phase sweep
                # increment_oscillator_phase=pulse.zhsweeper, # I believe this is the relative phase sweep
            )
        elif "frequency" in partial_sweep.uid or partial_sweep.uid == "start":
            exp.play(
                signal=f"{section}{qubit.name}",
                pulse=pulse.zhpulse,
                phase=pulse.pulse.relative_phase,
            )

    # Hardcoded for the flux pulse for 2q gates
    @staticmethod
    def play_sweep_select_dual(exp, qubit, pulse, section, parameters):
        """Play Zurich pulse when two sweepers are involved on the same
        pulse."""
        if "amplitude" in parameters and "duration" in parameters:
            for sweeper in pulse.zhsweepers:
                if sweeper.uid == "amplitude":
                    sweeper_amp_index = pulse.zhsweepers.index(sweeper)
                    sweeper.values = sweeper.values.copy()
                    pulse.zhpulse.amplitude *= max(abs(sweeper.values))
                    sweeper.values /= max(abs(sweeper.values))
                else:
                    sweeper_dur_index = pulse.zhsweepers.index(sweeper)

            exp.play(
                signal=f"{section}{qubit.name}",
                pulse=pulse.zhpulse,
                amplitude=pulse.zhsweepers[sweeper_amp_index],
                length=pulse.zhsweepers[sweeper_dur_index],
            )

    def play_sweep(self, exp, qubit, pulse, section):
        """Takes care of playing the sweepers and involved pulses for different
        options."""

        if isinstance(pulse, ZhSweeperLine):
            if pulse.zhsweeper.uid == "bias":
                exp.play(
                    signal=f"{section}{qubit.name}",
                    pulse=pulse.zhpulse,
                    amplitude=pulse.zhsweeper,
                )
        else:
            parameters = []
            for partial_sweep in pulse.zhsweepers:
                parameters.append(partial_sweep.uid)
            # Recheck partial sweeps
            if len(parameters) == 2:
                self.play_sweep_select_dual(exp, qubit, pulse, section, parameters)
            else:
                self.play_sweep_select_single(
                    exp, qubit, pulse, section, parameters, partial_sweep
                )

    def couplerflux(self, exp: lo.Experiment, couplers: Dict[str, Coupler]):
        """Coupler flux for bias sweep or pulses.

        Args:
            exp (lo.Experiment): laboneq experiment on which register sequences.
            couplers (dict[str, Coupler]): coupler on which pulses are played.
        """
        for coupler in couplers.values():
            c = coupler.name  # pylint: disable=C0103
            time = 0
            previous_section = None
            for i, sequence in enumerate(self.sub_sequences[f"couplerflux{c}"]):
                section_uid = f"sequence_couplerflux{c}_{i}"
                with exp.section(uid=section_uid, play_after=previous_section):
                    for j, pulse in enumerate(sequence):
                        pulse.zhpulse.uid += f"{i}_{j}"
                        exp.delay(
                            signal=f"couplerflux{c}",
                            time=round(pulse.pulse.start * NANO_TO_SECONDS, 9) - time,
                        )
                        time = round(pulse.pulse.duration * NANO_TO_SECONDS, 9) + round(
                            pulse.pulse.start * NANO_TO_SECONDS, 9
                        )
                        if isinstance(pulse, ZhSweeperLine):
                            self.play_sweep(exp, coupler, pulse, section="couplerflux")
                        elif isinstance(pulse, ZhSweeper):
                            self.play_sweep(exp, coupler, pulse, section="couplerflux")
                        elif isinstance(pulse, ZhPulse):
                            exp.play(signal=f"couplerflux{c}", pulse=pulse.zhpulse)
                previous_section = section_uid

    def flux(self, exp: lo.Experiment, qubits: Dict[str, Qubit]):
        """Qubit flux for bias sweep or pulses.

        Args:
            exp (lo.Experiment): laboneq experiment on which register sequences.
            qubits (dict[str, Qubit]): qubits on which pulses are played.
        """
        for qubit in qubits.values():
            q = qubit.name  # pylint: disable=C0103
            time = 0
            previous_section = None
            for i, sequence in enumerate(self.sub_sequences[f"flux{q}"]):
                section_uid = f"sequence_flux{q}_{i}"
                with exp.section(uid=section_uid, play_after=previous_section):
                    for j, pulse in enumerate(sequence):
                        if not isinstance(pulse, ZhSweeperLine):
                            pulse.zhpulse.uid += f"{i}_{j}"
                            exp.delay(
                                signal=f"flux{q}",
                                time=round(pulse.pulse.start * NANO_TO_SECONDS, 9)
                                - time,
                            )
                            time = round(
                                pulse.pulse.duration * NANO_TO_SECONDS, 9
                            ) + round(pulse.pulse.start * NANO_TO_SECONDS, 9)
                        if isinstance(pulse, ZhSweeperLine):
                            self.play_sweep(exp, qubit, pulse, section="flux")
                        elif isinstance(pulse, ZhSweeper):
                            self.play_sweep(exp, qubit, pulse, section="flux")
                        elif isinstance(pulse, ZhPulse):
                            exp.play(signal=f"flux{q}", pulse=pulse.zhpulse)
                previous_section = section_uid

    def drive(self, exp: lo.Experiment, qubits: Dict[str, Qubit]):
        """Qubit driving pulses.

        Args:
            exp (lo.Experiment): laboneq experiment on which register sequences.
            qubits (dict[str, Qubit]): qubits on which pulses are played.
        """
        for qubit in qubits.values():
            q = qubit.name  # pylint: disable=C0103
            time = 0
            previous_section = None
            for i, sequence in enumerate(self.sub_sequences[f"drive{q}"]):
                section_uid = f"sequence_drive{q}_{i}"
                with exp.section(uid=section_uid, play_after=previous_section):
                    for j, pulse in enumerate(sequence):
                        if not isinstance(pulse, ZhSweeperLine):
                            exp.delay(
                                signal=f"drive{q}",
                                time=round(pulse.pulse.start * NANO_TO_SECONDS, 9)
                                - time,
                            )
                            time = round(
                                pulse.pulse.duration * NANO_TO_SECONDS, 9
                            ) + round(pulse.pulse.start * NANO_TO_SECONDS, 9)
                            pulse.zhpulse.uid += f"{i}_{j}"
                            if isinstance(pulse, ZhSweeper):
                                self.play_sweep(exp, qubit, pulse, section="drive")
                            elif isinstance(pulse, ZhPulse):
                                exp.play(
                                    signal=f"drive{q}",
                                    pulse=pulse.zhpulse,
                                    phase=pulse.pulse.relative_phase,
                                )
                        elif isinstance(pulse, ZhSweeperLine):
                            exp.delay(signal=f"drive{q}", time=pulse.zhsweeper)

                    if len(self.sequence[f"readout{q}"]) > 0 and isinstance(
                        self.sequence[f"readout{q}"][0], ZhSweeperLine
                    ):
                        exp.delay(
                            signal=f"drive{q}",
                            time=self.sequence[f"readout{q}"][0].zhsweeper,
                        )
                        self.sequence[f"readout{q}"].remove(
                            self.sequence[f"readout{q}"][0]
                        )

                previous_section = section_uid

    def find_subsequence_finish(
        self,
        measurement_number: int,
        line: str,
        quantum_elements: Union[Dict[str, Qubit], Dict[str, Coupler]],
    ) -> Tuple[int, str]:
        """Find the finishing time and qubit for a given sequence.

        Args:
            measurement_number (int): number of the measure pulse.
            line (str): line from which measure the finishing time.
                e.g.: "drive", "flux", "couplerflux"
            quantum_elements (dict[str, Qubit]|dict[str, Coupler]): qubits or couplers from
                which measure the finishing time.

        Returns:
            time_finish (int): Finish time of the last pulse of the subsequence
                before the measurement.
            sequence_finish (str): Name of the last subsequence before measurement. If
                there are no sequences after the previous measurement, use "None".
        """
        time_finish = 0
        sequence_finish = "None"
        for quantum_element in quantum_elements:
            if (
                len(self.sub_sequences[f"{line}{quantum_element}"])
                <= measurement_number
            ):
                continue
            for pulse in self.sub_sequences[f"{line}{quantum_element}"][
                measurement_number
            ]:
                if pulse.pulse.finish > time_finish:
                    time_finish = pulse.pulse.finish
                    sequence_finish = f"{line}{quantum_element}"
        return time_finish, sequence_finish

    # For pulsed spectroscopy, set integration_length and either measure_pulse or measure_pulse_length.
    # For CW spectroscopy, set only integration_length and do not specify the measure signal.
    # For all other measurements, set either length or pulse for both the measure pulse and integration kernel.
    def measure_relax(self, exp, qubits, couplers, relaxation_time, acquisition_type):
        """Qubit readout pulse, data acquisition and qubit relaxation."""
        readout_schedule = defaultdict(list)
        qubit_readout_schedule = defaultdict(list)
        iq_angle_readout_schedule = defaultdict(list)
        for qubit in qubits.values():
            q = qubit.name  # pylint: disable=C0103
            iq_angle = qubit.iq_angle
            if len(self.sequence[f"readout{q}"]) != 0:
                for i, pulse in enumerate(self.sequence[f"readout{q}"]):
                    readout_schedule[i].append(pulse)
                    qubit_readout_schedule[i].append(q)
                    iq_angle_readout_schedule[i].append(iq_angle)

        weights = {}
        for i, (pulses, qubits_readout, iq_angles) in enumerate(
            zip(
                readout_schedule.values(),
                qubit_readout_schedule.values(),
                iq_angle_readout_schedule.values(),
            )
        ):
            qd_finish = self.find_subsequence_finish(i, "drive", qubits_readout)
            qf_finish = self.find_subsequence_finish(i, "flux", qubits_readout)
            cf_finish = self.find_subsequence_finish(i, "couplerflux", couplers)
            finish_times = np.array(
                [
                    qd_finish,
                    qf_finish,
                    cf_finish,
                ],
                dtype=[("finish", "i4"), ("line", "U15")],
            )
            latest_sequence = finish_times[finish_times["finish"].argmax()]
            if latest_sequence["line"] == "None":
                play_after = None
            else:
                play_after = f"sequence_{latest_sequence['line']}_{i}"
            # Section on the outside loop allows for multiplex
            with exp.section(uid=f"sequence_measure_{i}", play_after=play_after):
                for pulse, q, iq_angle in zip(pulses, qubits_readout, iq_angles):
                    pulse.zhpulse.uid += str(i)

                    exp.delay(
                        signal=f"acquire{q}",
                        time=self.smearing * NANO_TO_SECONDS,
                    )

                    if (
                        qubits[q].kernel is not None
                        and acquisition_type == lo.AcquisitionType.DISCRIMINATION
                    ):
                        kernel = qubits[q].kernel
                        weight = lo.pulse_library.sampled_pulse_complex(
                            uid="weight" + str(q),
                            samples=kernel * np.exp(1j * iq_angle),
                        )

                    else:
                        if i == 0:
                            if acquisition_type == lo.AcquisitionType.DISCRIMINATION:
                                weight = lo.pulse_library.sampled_pulse_complex(
                                    samples=np.ones(
                                        [
                                            int(
                                                pulse.pulse.duration * 2
                                                - 3 * self.smearing * NANO_TO_SECONDS
                                            )
                                        ]
                                    )
                                    * np.exp(1j * iq_angle),
                                    uid="weights" + str(q),
                                )
                                weights[q] = weight
                            else:
                                # TODO: Patch for multiple readouts: Remove different uids
                                weight = lo.pulse_library.const(
                                    uid="weight" + str(q),
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

                    if i == len(self.sequence[f"readout{q}"]) - 1:
                        reset_delay = relaxation_time * NANO_TO_SECONDS
                    else:
                        # Here time of flight or not ?
                        reset_delay = 0  # self.time_of_flight * NANO_TO_SECONDS

                    exp.measure(
                        acquire_signal=f"acquire{q}",
                        handle=f"sequence{q}_{i}",
                        integration_kernel=weight,
                        integration_kernel_parameters=None,
                        integration_length=None,
                        measure_signal=f"measure{q}",
                        measure_pulse=pulse.zhpulse,
                        measure_pulse_length=round(
                            pulse.pulse.duration * NANO_TO_SECONDS, 9
                        ),
                        measure_pulse_parameters=measure_pulse_parameters,
                        measure_pulse_amplitude=None,
                        acquire_delay=self.time_of_flight * NANO_TO_SECONDS,
                        reset_delay=reset_delay,
                    )

    def fast_reset(self, exp, qubits, fast_reset):
        """
        Conditional fast reset after readout - small delay for signal processing
        This is a very naive approach that can be improved by repeating this step until
        we reach non fast reset fidelity
        https://quantum-computing.ibm.com/lab/docs/iql/manage/systems/reset/backend_reset
        """
        log.warning("Im fast resetting")
        for qubit_name in self.sequence_qibo.qubits:
            qubit = qubits[qubit_name]
            q = qubit.name  # pylint: disable=C0103
            with exp.section(uid=f"fast_reset{q}", play_after=f"sequence_measure"):
                with exp.match_local(handle=f"sequence{q}"):
                    with exp.case(state=0):
                        pass
                    with exp.case(state=1):
                        pulse = ZhPulse(qubit.native_gates.RX.pulse(0, 0))
                        exp.play(signal=f"drive{q}", pulse=pulse.zhpulse)

    @staticmethod
    def rearrange_sweepers(sweepers: List[Sweeper]) -> Tuple[np.ndarray, List[Sweeper]]:
        """Rearranges sweepers from qibocal based on device hardware
        limitations.

        Frequency sweepers must be applied before (on the outer loop) bias or amplitude sweepers.

        Args:
            sweepers (list[Sweeper]): list of sweepers used in the experiment.

        Returns:
            rearranging_axes (np.ndarray): array of shape (2,) and dtype=int containing
                the indexes of the sweepers to be swapped. Defaults to np.array([0, 0])
                if no swap is needed.
            sweepers (list[Sweeper]): updated list of sweepers used in the experiment. If
                sweepers must be swapped, the list is updated accordingly.
        """
        rearranging_axes = np.zeros(2, dtype=int)
        if len(sweepers) == 2:
            if sweepers[1].parameter is Parameter.frequency:
                if not sweepers[0].pulses is None:
                    if (sweepers[0].parameter is Parameter.bias) or (
                        not sweepers[0].parameter is Parameter.amplitude
                        and sweepers[0].pulses[0].type is not PulseType.READOUT
                    ):
                        rearranging_axes[:] = [1, 0]
                        sweepers = sweepers[::-1]
                        log.warning("Sweepers were reordered")
        return rearranging_axes, sweepers

    def sweep(self, qubits, couplers, sequence: PulseSequence, options, *sweepers):
        """Play pulse and sweepers sequence."""

        self.signal_map = {}
        self.nt_sweeps = None
        sweepers = list(sweepers)
        rearranging_axes, sweepers = self.rearrange_sweepers(sweepers)
        self.sweepers = sweepers

        self.frequency_from_pulses(qubits, sequence)

        self.experiment_flow(qubits, couplers, sequence, options)
        self.run_exp()

        #  Get the results back
        results = {}
        for qubit in qubits.values():
            q = qubit.name  # pylint: disable=C0103
            if len(self.sequence[f"readout{q}"]) != 0:
                for i, ropulse in enumerate(self.sequence[f"readout{q}"]):
                    exp_res = self.results.get_data(f"sequence{q}_{i}")
                    # if using singleshot, the first axis contains shots,
                    # i.e.: (nshots, sweeper_1, sweeper_2)
                    # if using integration: (sweeper_1, sweeper_2)
                    if options.averaging_mode is AveragingMode.SINGLESHOT:
                        rearranging_axes += 1
                    # Reorder dimensions
                    data = np.moveaxis(
                        exp_res, rearranging_axes[0], rearranging_axes[1]
                    )
                    if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                        data = (
                            np.ones(data.shape) - data.real
                        )  # Probability inversion patch

                    serial = ropulse.pulse.serial
                    qubit = ropulse.pulse.qubit
                    results[serial] = results[qubit] = options.results_type(data)

        # html containing the pulse sequence schedule
        # lo.show_pulse_sheet("pulses", self.exp)
        return results

    def sweep_recursion(self, qubits, couplers, exp, exp_calib, exp_options):
        """Sweepers recursion for multiple nested Real Time sweepers."""

        sweeper = self.sweepers[0]

        i = len(self.sweepers) - 1
        self.sweepers.remove(sweeper)
        parameter = None

        if sweeper.parameter is Parameter.frequency:
            for pulse in sweeper.pulses:
                line = "drive" if pulse.type is PulseType.DRIVE else "measure"
                zhsweeper = ZhSweeper(
                    pulse, sweeper, qubits[sweeper.pulses[0].qubit]
                ).zhsweeper
                zhsweeper.uid = "frequency"  # Changing the name from "frequency" breaks it f"frequency_{i}
                exp_calib[f"{line}{pulse.qubit}"] = lo.SignalCalibration(
                    oscillator=lo.Oscillator(
                        frequency=zhsweeper,
                        modulation_type=lo.ModulationType.HARDWARE,
                    )
                )
        if sweeper.parameter is Parameter.amplitude:
            for pulse in sweeper.pulses:
                pulse = pulse.copy()
                pulse.amplitude *= max(abs(sweeper.values))

                # Proper copy(sweeper) here if we want to keep the sweepers
                # sweeper_aux = copy.copy(sweeper)
                aux_max = max(abs(sweeper.values))

                sweeper.values /= aux_max
                parameter = ZhSweeper(
                    pulse, sweeper, qubits[sweeper.pulses[0].qubit]
                ).zhsweeper
                sweeper.values *= aux_max

        if sweeper.parameter is Parameter.bias:
            if sweeper.qubits:
                for qubit in sweeper.qubits:
                    parameter = ZhSweeperLine(
                        sweeper, qubit, self.sequence_qibo
                    ).zhsweeper
            if sweeper.couplers:
                for qubit in sweeper.couplers:
                    parameter = ZhSweeperLine(
                        sweeper, qubit, self.sequence_qibo
                    ).zhsweeper

        elif sweeper.parameter is Parameter.start:
            parameter = ZhSweeperLine(sweeper).zhsweeper

        elif parameter is None:
            parameter = ZhSweeper(
                sweeper.pulses[0], sweeper, qubits[sweeper.pulses[0].qubit]
            ).zhsweeper

        with exp.sweep(
            uid=f"sweep_{sweeper.parameter.name.lower()}_{i}",  # This uid trouble double freq ???
            parameter=parameter,
            reset_oscillator_phase=True,  # Should we reset this phase ???
        ):
            if len(self.sweepers) > 0:
                self.sweep_recursion(qubits, couplers, exp, exp_calib, exp_options)
            else:
                self.select_exp(exp, qubits, couplers, exp_options)

    def find_instrument_address(
        self, quantum_element: Union[Qubit, Coupler], parameter: str
    ) -> str:
        """Find path of the instrument connected to a specified line and
        qubit/coupler.

        Args:
            quantum_element (Qubit | Coupler): qubits or couplers on which perform the near time sweep.
            parameter (str): parameter on which perform the near time sweep.
        """
        line_names = {
            "bias": "flux",
            "amplitude": "drive",
        }
        line_name = line_names[parameter]
        channel_uid = (
            self.device_setup.logical_signal_groups[f"q{quantum_element.name}"]
            .logical_signals[f"{line_name}_line"]
            .physical_channel.uid
        )
        channel_name = channel_uid.split("/")[0]
        instruments = self.device_setup.instruments
        for instrument in instruments:
            if instrument.uid == channel_name:
                return instrument.address
        raise RuntimeError(
            f"Could not find instrument for {quantum_element} {line_name}"
        )

    def sweep_recursion_nt(
        self,
        qubits: Dict[str, Qubit],
        couplers: Dict[str, Coupler],
        options: ExecutionParameters,
        exp: lo.Experiment,
        exp_calib: lo.Calibration,
    ):
        """Sweepers recursion for Near Time sweepers. Faster than regular
        software sweepers as they are executed on the actual device by
        (software ? or slower hardware ones)

        You want to avoid them so for now they are implement for a
        specific sweep.
        """

        log.info("nt Loop")

        sweeper = self.nt_sweeps[0]

        i = len(self.nt_sweeps) - 1
        self.nt_sweeps.remove(sweeper)

        parameter = None

        if sweeper.parameter is Parameter.bias:
            if sweeper.qubits:
                for qubit in sweeper.qubits:
                    zhsweeper = ZhSweeperLine(
                        sweeper, qubit, self.sequence_qibo
                    ).zhsweeper
                    zhsweeper.uid = "bias"
                    path = self.find_instrument_address(qubit, "bias")

                    parameter = copy.deepcopy(zhsweeper)
                    parameter.values += qubit.flux.offset
                    device_path = f"{path}/sigouts/0/offset"

        elif sweeper.parameter is Parameter.amplitude:
            for pulse in sweeper.pulses:
                pulse = pulse.copy()
                pulse.amplitude *= max(abs(sweeper.values))

                # Proper copy(sweeper) here
                # sweeper_aux = copy.copy(sweeper)
                aux_max = max(abs(sweeper.values))

                sweeper.values /= aux_max
                zhsweeper = ZhSweeper(
                    pulse, sweeper, qubits[sweeper.pulses[0].qubit]
                ).zhsweeper
                sweeper.values *= aux_max

                zhsweeper.uid = "amplitude"
                path = self.find_instrument_address(
                    qubits[sweeper.pulses[0].qubit], "amplitude"
                )
                parameter = zhsweeper
                device_path = (
                    f"/{path}/qachannels/*/oscs/0/gain"  # Hardcoded SHFQA device
                )

        elif parameter is None:  # can it be accessed?
            parameter = ZhSweeper(
                sweeper.pulses[0], sweeper, qubits[sweeper.pulses[0].qubit]
            ).zhsweeper
            device_path = f"/{path}/qachannels/*/oscs/0/gain"  # Hardcoded SHFQA device

        with exp.sweep(
            uid=f"sweep_{sweeper.parameter.name.lower()}_{i}",
            parameter=parameter,
        ):
            exp.set_node(
                path=device_path,
                value=parameter,
            )

            if len(self.nt_sweeps) > 0:
                self.sweep_recursion_nt(qubits, couplers, options, exp, exp_calib)
            else:
                self.define_exp(qubits, couplers, options, exp, exp_calib)

    def play_sim(self, qubits, sequence, options, sim_time):
        """Play pulse sequence."""

        self.experiment_flow(qubits, sequence, options)  # missing couplers?
        self.run_sim(sim_time)

    def run_sim(self, sim_time):
        """Run the simulation.

        Args:
            sim_time (float): Time[s] to simulate starting from 0
        """
        # create a session
        self.sim_session = lo.Session(self.device_setup)
        # connect to session
        self.sim_device = self.sim_session.connect(do_emulation=True)
        self.exp = self.sim_session.compile(
            self.experiment, compiler_settings=COMPILER_SETTINGS
        )

        # Plot simulated output signals with helper function
        plot_simulation(
            self.exp,
            start_time=0,
            length=sim_time,
            plot_width=10,
            plot_height=3,
        )
