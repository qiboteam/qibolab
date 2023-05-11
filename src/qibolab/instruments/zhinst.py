import os
from collections import defaultdict
from pathlib import Path

import laboneq._token
import laboneq.simple as lo
import numpy as np

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.paths import qibolab_folder
from qibolab.platforms.platform import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.pulses import FluxPulse, PulseSequence, PulseType
from qibolab.result import ExecutionResults
from qibolab.sweeper import Parameter

os.environ["LABONEQ_TOKEN"] = "ciao come va?"  # or other random text
laboneq._token.is_valid_token = lambda _token: True

# FIXME: Multiplex (For readout). Workaround integration weights padding with zeros.
# FIXME: Handle on acquires for list of pulse sequences
# FIXME: I think is a hardware limitation but I cant sweep multiple drive oscillator at the same time


class ZhPulse:
    """Zurich pulse from qibolab pulse"""

    def __init__(self, pulse):
        """Qibolab pulse"""
        self.pulse = pulse
        """Line associated with the pulse"""
        self.signal = f"{pulse.type.name.lower()}{pulse.qubit}"
        """Zurich pulse"""
        self.zhpulse = self.select_pulse(pulse, pulse.type.name.lower())

    # FIXME: Either implement more or create and arbitrary one
    def select_pulse(self, pulse, pulse_type):
        """Pulse translation"""

        if str(pulse.shape) == "Rectangular()":
            zh_pulse = lo.pulse_library.const(
                uid=(f"{pulse_type}_{pulse.qubit}_"),
                length=round(pulse.duration * 1e-9, 9),
                amplitude=pulse.amplitude,
            )
        elif "Gaussian" in str(pulse.shape):
            sigma = pulse.shape.rel_sigma
            zh_pulse = lo.pulse_library.gaussian(
                uid=(f"{pulse_type}_{pulse.qubit}_"),
                length=round(pulse.duration * 1e-9, 9),
                amplitude=pulse.amplitude,
                sigma=2 / sigma,
                zero_boundaries=False,
            )

        elif "GaussianSquare" in str(pulse.shape):
            sigma = pulse.shape.rel_sigma
            zh_pulse = lo.pulse_library.gaussian_square(
                uid=(f"{pulse_type}_{pulse.qubit}_"),
                length=round(pulse.duration * 1e-9, 9),
                width=round(pulse.duration * 1e-9, 9) * 0.9,  # 90% Flat
                amplitude=pulse.amplitude,
                sigma=2 / sigma,
                zero_boundaries=False,
            )

        elif "Drag" in str(pulse.shape):
            sigma = pulse.shape.rel_sigma
            beta = pulse.shape.beta
            zh_pulse = lo.pulse_library.drag(
                uid=(f"{pulse_type}_{pulse.qubit}_"),
                length=round(pulse.duration * 1e-9, 9),
                amplitude=pulse.amplitude,
                sigma=2 / sigma,
                beta=beta,
                zero_boundaries=False,
            )
        elif "Slepian" in str(pulse.shape):
            "Implement Slepian shaped flux pulse https://arxiv.org/pdf/0909.5368.pdf"

        elif "Sampled" in str(pulse.shape):
            "Implement Sampled pulses for Optimal control algorithms like GRAPE"

            """
            Typically, the sampler function should discard ``length`` and ``amplitude``, and
            instead assume that the pulse extends from -1 to 1, and that it has unit
            amplitude. LabOne Q will automatically rescale the sampler's output to the correct
            amplitude and length.

            They don't even do that on their notebooks
            and just use lenght and amplitude but we have to check

            x = pulse.envelope_waveform_i.data  No need for q ???
            """

        return zh_pulse


class ZhSweeper:
    """
    Zurich sweeper from qibolab sweeper for pulse parameters
    Amplitude, Duration, Frequency (and maybe Phase)

    """

    def __init__(self, pulse, sweeper, qubit):
        """Qibolab sweeper"""
        self.sweeper = sweeper

        """Qibolab pulse associated to the sweeper"""
        self.pulse = pulse
        """Line associated with the pulse"""
        self.signal = f"{pulse.type.name.lower()}{pulse.qubit}"
        """Zurich pulse associated to the sweeper"""
        self.zhpulse = ZhPulse(pulse).zhpulse

        """Zurich sweeper"""
        self.zhsweeper = self.select_sweeper(sweeper, qubit)

        """
        Zurich sweepers, Need something better to store multiple sweeps on the same pulse

        Not properly implemented as it was only used on Rabi amplitude vs lenght and it
        was an unused routine.
        """
        self.zhsweepers = [self.select_sweeper(sweeper, qubit)]

    def select_sweeper(self, sweeper, qubit):
        """Sweeper translation"""

        if sweeper.parameter is Parameter.amplitude:
            zh_sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values,
            )
        elif sweeper.parameter is Parameter.duration:
            zh_sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values * 1e-9,
            )
        # TODO: take intermediate frequency from the pulses
        elif sweeper.parameter is Parameter.frequency:
            if self.pulse.type is PulseType.READOUT:
                intermediate_frequency = qubit.readout_frequency - qubit.readout.local_oscillator.frequency
            elif self.pulse.type is PulseType.DRIVE:
                intermediate_frequency = qubit.drive_frequency - qubit.drive.local_oscillator.frequency
            zh_sweeper = lo.LinearSweepParameter(
                uid=sweeper.parameter.name,
                start=sweeper.values[0] + intermediate_frequency,
                stop=sweeper.values[-1] + intermediate_frequency,
                count=len(sweeper.values),
            )
        return zh_sweeper

    def add_sweeper(self, sweeper, qubit):
        """Add sweeper to list of sweepers"""
        self.zhsweepers.append(self.select_sweeper(sweeper, qubit))


class ZhSweeperLine:
    """
    Zurich sweeper from qibolab sweeper for non pulse parameters
    Bias, Delay (, power_range, local_oscillator frequency, offset ???)

    For now Parameter.bias sweepers are implemented as Parameter.Amplitude
    on a flux pulse. We may want to keep this class separate for future
    Near Time sweeps
    """

    def __init__(self, sweeper, qubit, sequence, pulse=None):
        """Qibolab sweeper"""
        self.sweeper = sweeper

        # TODO: I already created a flux pulse check
        if sweeper.parameter is Parameter.bias:
            self.pulse = FluxPulse(
                start=sequence.start,
                duration=sequence.duration,
                amplitude=1,
                shape="Rectangular",
                channel=qubit.flux.name,
                qubit=qubit.name,
            )
            self.signal = f"flux{qubit.name}"

            self.zh_pulse = lo.pulse_library.const(
                uid=(f"{pulse.type.name.lower()}_{pulse.qubit}_"),
                length=round(pulse.duration * 1e-9, 9),
                amplitude=pulse.amplitude,
            )

        # Need something better to store multiple sweeps on the same pulse
        self.zhsweeper = self.select_sweeper(sweeper)

    def select_sweeper(self, sweeper):
        """Sweeper translation"""
        if sweeper.parameter is Parameter.bias:
            zh_sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values,
            )
        if sweeper.parameter is Parameter.delay:
            zh_sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values * 1e-9,
            )
        return zh_sweeper


class Zurich(AbstractInstrument):
    """Zurich driver main class"""

    def __init__(self, name, descriptor, use_emulation=False):
        "Setup name (str)"
        self.name = name
        """
        Port and device mapping in yaml text (str)

        It should be used as a template by adding extra lines for each of the different
        frequency pulses played thought the same port after parsing the sequence.
        """
        self.descriptor = descriptor
        "Enable emulation mode (bool)"
        self.emulation = use_emulation

        "Is the device connected ? (bool)"
        self.is_connected = False

        "Signals to lines mapping"
        self.signal_map = {}
        "Zurich calibration object)"
        self.calibration = lo.Calibration()

        "Zurich device parameters for connection"
        self.device_setup = None
        self.session = None
        self.device = None

        "Parameters read from the runcard not part of ExecutionParameters"
        self.Fast_reset = False
        self.time_of_flight = 0.0
        self.smearing = 0.0
        self.chip = "iqm5q"

        "Zurich experiment definitions"
        self.exp = None
        self.experiment = None
        self.exp_options = ExecutionParameters
        self.exp_calib = lo.Calibration()
        self.results = None

        "To store if the AcquisitionType.SPECTROSCOPY needs to be enabled by parsing the sequence"
        self.acquisition_type = None

        "Zurich pulse sequence"
        self.sequence = defaultdict(list)
        # Remove if able
        self.sequence_qibo = None

        # Improve the storing of multiple sweeps
        "Storing sweepers"
        self.sweepers = []
        self.NT_sweeps = []

    def connect(self):
        if not self.is_connected:
            for _ in range(3):
                try:
                    self.device_setup = lo.DeviceSetup.from_descriptor(
                        yaml_text=self.descriptor,
                        server_host="localhost",
                        server_port="8004",
                        setup_name=self.name,
                    )
                    self.session = lo.Session(self.device_setup)
                    self.device = self.session.connect(do_emulation=self.emulation)
                    self.is_connected = True
                    break
                except Exception as exc:
                    print(f"Unable to connect:\n{str(exc)}\nRetrying...")
            if not self.is_connected:
                raise InstrumentException(self, f"Unable to connect to {self.name}")

    def start(self):
        pass

    def stop(self):
        pass

    def disconnect(self):
        if self.is_connected:
            self.device = self.session.disconnect()
            self.is_connected = False
        else:
            print("Already disconnected")

    # FIXME: Not working so it does not get the settings
    def setup(self, **_kwargs):
        pass

    def calibration_step(self, qubits):
        """
        Zurich general pre experiment calibration definitions

        Change to get frequencies from sequence
        """

        for qubit in qubits.values():
            if qubit.flux_coupler:
                self.register_flux_line(qubit)
            else:
                if qubit.flux is not None:
                    self.register_flux_line(qubit)
                if self.sequence[f"drive{qubit.name}"]:
                    self.register_drive_line(
                        qubit=qubit,
                        intermediate_frequency=qubit.drive_frequency - qubit.drive.local_oscillator.frequency,
                    )
                if self.sequence[f"readout{qubit.name}"]:
                    self.register_readout_line(
                        qubit=qubit,
                        intermediate_frequency=qubit.readout_frequency - qubit.readout.local_oscillator.frequency,
                    )
        self.device_setup.set_calibration(self.calibration)

    def register_readout_line(self, qubit, intermediate_frequency):
        """Registers qubit measure and acquire lines to calibration and signal map."""

        q = qubit.name
        self.signal_map[f"measure{q}"] = self.device_setup.logical_signal_groups[f"q{q}"].logical_signals[
            "measure_line"
        ]
        self.calibration[f"/logical_signal_groups/q{q}/measure_line"] = lo.SignalCalibration(
            oscillator=lo.Oscillator(
                frequency=intermediate_frequency,
                modulation_type=lo.ModulationType.SOFTWARE,
            ),
            local_oscillator=lo.Oscillator(
                frequency=int(qubit.readout.local_oscillator.frequency),
                # frequency=0.0, # This and PortMode.LF allow debugging with and oscilloscope
            ),
            range=qubit.readout.power_range,
            port_delay=None,
            # port_mode= lo.PortMode.LF,
            delay_signal=0,
        )

        self.signal_map[f"acquire{q}"] = self.device_setup.logical_signal_groups[f"q{q}"].logical_signals[
            "acquire_line"
        ]
        self.calibration[f"/logical_signal_groups/q{q}/acquire_line"] = lo.SignalCalibration(
            oscillator=lo.Oscillator(
                frequency=intermediate_frequency,
                modulation_type=lo.ModulationType.SOFTWARE,
            ),
            local_oscillator=lo.Oscillator(
                frequency=int(qubit.readout.local_oscillator.frequency),
            ),
            range=qubit.feedback.power_range,
            port_delay=self.time_of_flight,
            threshold=qubit.threshold,
        )

    def register_drive_line(self, qubit, intermediate_frequency):
        """Registers qubit drive line to calibration and signal map."""
        q = qubit.name
        self.signal_map[f"drive{q}"] = self.device_setup.logical_signal_groups[f"q{q}"].logical_signals["drive_line"]
        self.calibration[f"/logical_signal_groups/q{q}/drive_line"] = lo.SignalCalibration(
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

    def register_flux_line(self, qubit):
        """Registers qubit flux line to calibration and signal map."""
        q = qubit.name
        self.signal_map[f"flux{q}"] = self.device_setup.logical_signal_groups[f"q{q}"].logical_signals["flux_line"]
        self.calibration[f"/logical_signal_groups/q{q}/flux_line"] = lo.SignalCalibration(
            range=qubit.flux.power_range, port_delay=None, delay_signal=0, voltage_offset=qubit.flux.bias
        )

    def run_exp(self):
        """Compilation settings, compilation step, execution step and data retrival"""
        compiler_settings = {
            "SHFSG_FORCE_COMMAND_TABLE": True,
            "SHFSG_MIN_PLAYWAVE_HINT": 32,
            "SHFSG_MIN_PLAYZERO_HINT": 32,
        }

        self.exp = self.session.compile(self.experiment, compiler_settings=compiler_settings)
        self.results = self.session.run(self.exp)

    # TODO: Play taking a big sequence with several acquire steps
    def play(self, qubits, sequence, options):
        """Play pulse sequence"""
        if options.relaxation_time is None:
            options.relaxation_time = self.relaxation_time

        # TODO: Read frequency for pulses instead of qubit patch
        for qubit in qubits.values():
            for pulse in sequence:
                if pulse.qubit == qubit.name:
                    if pulse.type is PulseType.READOUT:
                        qubit.readout_frequency = pulse.frequency
                    if pulse.type is PulseType.DRIVE:
                        qubit.drive_frequency = pulse.frequency

        """
        Play pulse sequence steps, one on each method:
        Translation, Calibration, Experiment Definition and Execution.
        """
        self.sequence_zh(sequence, qubits, sweepers=[])
        self.calibration_step(qubits)
        self.create_exp(qubits, options)
        self.run_exp()

        # TODO: General, several readouts and qubits
        # TODO: Implement the new results!
        "Get the results back"
        results = {}
        for qubit in qubits.values():
            if not qubit.flux_coupler:
                if self.sequence[f"readout{qubit.name}"]:
                    exp_res = self.results.get_data(f"sequence{qubit.name}")
                    i = exp_res.real
                    q = exp_res.imag
                    results[self.sequence[f"readout{qubit.name}"][0].pulse.serial] = ExecutionResults.from_components(
                        i, q
                    )

        # FIXME: Include this on the reports
        # html containing the pulse sequence schedule
        # lo.show_pulse_sheet("pulses", self.exp)

        # There is no reason for disconnection and it prevents reconnection
        # for a period of time making the software loops with execute_play_sequence crash
        # self.disconnect()
        return results

    def sequence_zh(self, sequence, qubits, sweepers):
        """Qibo sequence to Zurich sequence"""

        "Define and assign the sequence"
        zhsequence = defaultdict(list)
        self.sequence_qibo = sequence

        "Fill the sequences with pulses according to their lines in temporal order"
        for pulse in sequence:
            zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"].append(ZhPulse(pulse))

        "Mess that gets the sweeper and substitutes the pulse it sweeps in the right place"
        for sweeper in sweepers:
            if sweeper.parameter.name in {"amplitude", "frequency", "duration", "relative_phase"}:
                for pulse in sweeper.pulses:
                    aux_list = zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"]
                    if sweeper.parameter is Parameter.frequency and pulse.type is PulseType.READOUT:
                        self.acquisition_type = lo.AcquisitionType.SPECTROSCOPY
                    if sweeper.parameter is Parameter.amplitude and pulse.type is PulseType.READOUT:
                        self.acquisition_type = lo.AcquisitionType.SPECTROSCOPY
                        self.NT_sweeps.append(sweeper)
                        self.sweepers.remove(sweeper)
                    for element in aux_list:
                        if pulse == element.pulse:
                            if isinstance(aux_list[aux_list.index(element)], ZhPulse):
                                aux_list[aux_list.index(element)] = ZhSweeper(pulse, sweeper, qubits[pulse.qubit])
                            elif isinstance(aux_list[aux_list.index(element)], ZhSweeper):
                                aux_list[aux_list.index(element)].add_sweeper(sweeper, qubits[pulse.qubit])
            elif sweeper.parameter.name in {"bias"}:
                for qubit in sweeper.qubits.values():
                    zhsequence[f"flux{qubit.name}"] = [ZhSweeperLine(sweeper, qubit, sequence)]
            # FIXME: This may not place the Zhsweeper when the delay occurs among different sections or lines
            elif sweeper.parameter.name in {"delay"}:
                pulse = sweeper.pulses[0]
                aux_list = zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"]
                for element in aux_list:
                    if isinstance(element, ZhPulse):
                        if pulse == element.pulse:
                            if isinstance(aux_list[aux_list.index(element)], ZhPulse):
                                aux_list.insert(
                                    aux_list.index(element) + 1, ZhSweeperLine(sweeper, pulse.qubit, sequence)
                                )

        self.sequence = zhsequence

    def create_exp(self, qubits, options):
        """Zurich experiment definition usig their Experiment class"""
        signals = []
        for qubit in qubits.values():
            if qubit.flux_coupler:
                signals.append(lo.ExperimentSignal(f"flux{qubit.name}"))
            else:
                if self.sequence[f"drive{qubit.name}"]:
                    signals.append(lo.ExperimentSignal(f"drive{qubit.name}"))
                if qubit.flux is not None:
                    signals.append(lo.ExperimentSignal(f"flux{qubit.name}"))
                if self.sequence[f"readout{qubit.name}"]:
                    signals.append(lo.ExperimentSignal(f"measure{qubit.name}"))
                    signals.append(lo.ExperimentSignal(f"acquire{qubit.name}"))

        exp = lo.Experiment(
            uid="Sequence",
            signals=signals,
        )

        ACQUISITION_TYPE = {
            AcquisitionType.INTEGRATION: lo.AcquisitionType.INTEGRATION,
            AcquisitionType.RAW: lo.AcquisitionType.RAW,
            AcquisitionType.DISCRIMINATION: lo.AcquisitionType.DISCRIMINATION,
        }

        AVERAGING_MODE = {
            AveragingMode.CYCLIC: lo.AveragingMode.CYCLIC,
            AveragingMode.SINGLESHOT: lo.AveragingMode.SINGLE_SHOT,
        }

        if self.acquisition_type:
            acquisition_type = self.acquisition_type
        else:
            acquisition_type = ACQUISITION_TYPE[options.acquisition_type]

        exp_options = ExecutionParameters(
            nshots=options.nshots,
            relaxation_time=options.relaxation_time,
            fast_reset=options.fast_reset,
            acquisition_type=acquisition_type,
            averaging_mode=AVERAGING_MODE[options.averaging_mode],
        )

        print(exp_options.acquisition_type)
        print(exp_options.averaging_mode)

        exp_calib = lo.Calibration()
        if self.NT_sweeps:
            self.sweep_recursion_NT(qubits, exp_options, exp, exp_calib)
        else:
            self.define_exp(qubits, exp_options, exp, exp_calib)

    def define_exp(self, qubits, options, exp, exp_calib):
        with exp.acquire_loop_rt(
            uid="shots",
            count=options.nshots,
            acquisition_type=options.acquisition_type,
            averaging_mode=options.averaging_mode,
        ):
            if self.sweepers:
                self.sweep_recursion(
                    qubits, exp, exp_calib, options.relaxation_time, options.acquisition_type, options.fast_reset
                )
            else:
                self.select_exp(exp, qubits, options.relaxation_time, options.acquisition_type, options.fast_reset)
            exp.set_calibration(exp_calib)
            exp.set_signal_map(self.signal_map)
            self.experiment = exp

    def select_exp(self, exp, qubits, relaxation_time, acquisition_type, fast_reset=False):
        """Build Zurich Experiment selecting the relevant sections"""
        if "drive" in str(self.sequence):
            if "flux" in str(self.sequence):
                self.flux(exp, qubits)
                self.drive(exp, qubits)
            else:
                self.drive(exp, qubits)
        elif "flux" in str(self.sequence):
            self.flux(exp, qubits)
        self.measure_relax(exp, qubits, relaxation_time, acquisition_type)
        if fast_reset is not False:
            self.fast_reset(exp, qubits, fast_reset)

    def play_sweep(self, exp, qubit, pulse, section):
        """Play Zurich pulse when a sweeper is involved"""
        # FIXME: This loop for when a pulse is swept with several parameters(Max:3[Lenght, Amplitude, Phase]?)
        if self.sweepers == "2 sweeps on one single pulse":  # Need a better way of checking
            exp.play(
                signal=f"{section}{qubit.name}",
                pulse=pulse.zhpulse,
                amplitude=pulse.zhsweepers[0],
                length=pulse.zhsweepers[1],
                phase=pulse.pulse.relative_phase,
            )

        elif isinstance(pulse, ZhSweeperLine):
            parameters = pulse.zhsweeper.uid
            if parameters == "bias":
                # if any("bias" in param for param in parameters):
                exp.play(
                    signal=f"{section}{qubit.name}",
                    pulse=pulse.zhpulse,
                    amplitude=pulse.zhsweeper,
                )

        else:
            parameters = []
            for partial_sweep in pulse.zhsweepers:
                parameters.append(partial_sweep.uid)
            if any("amplitude" in param for param in parameters):
                # Zurich is already multiplying the pulse amplitude with the sweeper amplitude
                pulse.zhpulse.amplitude = 1
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
            elif any("bias" in param for param in parameters):
                exp.play(
                    signal=f"{section}{qubit.name}",
                    pulse=pulse.zhpulse,
                    amplitude=pulse.zhsweeper,
                    phase=pulse.pulse.relative_phase,
                )
            elif "frequency" in partial_sweep.uid or partial_sweep.uid == "delay":
                exp.play(
                    signal=f"{section}{qubit.name}",
                    pulse=pulse.zhpulse,
                    phase=pulse.pulse.relative_phase,
                )

    def flux(self, exp, qubits):
        """qubit flux or qubit coupler flux for bias or pulses"""
        for qubit in qubits.values():
            with exp.section(uid=f"sequence_bias{qubit.name}"):
                i = 0
                time = 0
                for pulse in self.sequence[f"flux{qubit.name}"]:
                    if not isinstance(pulse, ZhSweeperLine):
                        pulse.zhpulse.uid = pulse.zhpulse.uid + str(i)
                        exp.delay(signal=f"flux{qubit.name}", time=round(pulse.pulse.start * 1e-9, 9) - time)
                        time += round(pulse.pulse.duration * 1e-9, 9) + round(pulse.pulse.start * 1e-9, 9) - time
                    if isinstance(pulse, ZhSweeperLine):
                        self.play_sweep(exp, qubit, pulse, section="flux")
                        print(qubit.name, pulse.zhpulse)
                    else:
                        exp.play(signal=f"flux{qubit.name}", pulse=pulse.zhpulse)
                    i += 1

    def drive(self, exp, qubits):
        """qubit driving pulses"""
        for qubit in qubits.values():
            if not qubit.flux_coupler:
                time = 0
                i = 0
                if self.sequence[f"drive{qubit.name}"]:
                    with exp.section(uid=f"sequence_drive{qubit.name}"):
                        for pulse in self.sequence[f"drive{qubit.name}"]:
                            # FIXME: Check delay schedule
                            if not isinstance(pulse, ZhSweeperLine):
                                exp.delay(signal=f"drive{qubit.name}", time=round(pulse.pulse.start * 1e-9, 9) - time)
                                time += (
                                    round(pulse.pulse.duration * 1e-9, 9) + round(pulse.pulse.start * 1e-9, 9) - time
                                )
                                pulse.zhpulse.uid = pulse.zhpulse.uid + str(i)
                                if isinstance(pulse, ZhSweeper):
                                    self.play_sweep(exp, qubit, pulse, section="drive")
                                elif isinstance(pulse, ZhPulse):
                                    exp.play(
                                        signal=f"drive{qubit.name}",
                                        pulse=pulse.zhpulse,
                                        phase=pulse.pulse.relative_phase,
                                    )
                                    i += 1
                            elif isinstance(pulse, ZhSweeperLine):
                                exp.delay(signal=f"drive{qubit.name}", time=pulse.zhsweeper)

    # For pulsed spectroscopy, set integration_length and either measure_pulse or measure_pulse_length.
    # For CW spectroscopy, set only integration_length and do not specify the measure signal.
    # For all other measurements, set either length or pulse for both the measure pulse and integration kernel.
    def measure_relax(self, exp, qubits, relaxation_time, acquisition_type):
        """qubit readout pulse, data acquisition and qubit relaxation to ground state"""
        for qubit in qubits.values():
            if not qubit.flux_coupler:
                play_after = None
                if self.sequence[f"drive{qubit.name}"]:
                    play_after = f"sequence_drive{qubit.name}"
                with exp.section(uid=f"sequence_measure{qubit.name}", play_after=play_after):
                    if self.sequence[f"drive{qubit.name}"]:
                        last_drive_pulse = self.sequence[f"drive{qubit.name}"][-1]
                        if isinstance(last_drive_pulse, ZhPulse):
                            time = round(last_drive_pulse.pulse.finish * 1e-9, 9)
                        else:
                            time = 0
                    else:
                        time = 0
                    i = 0
                    if self.sequence[f"readout{qubit.name}"]:
                        for pulse in self.sequence[f"readout{qubit.name}"]:
                            exp.delay(signal=f"measure{qubit.name}", time=0)
                            # FIXME: This part
                            # exp.delay(signal=f"measure{qubit.name}", time=round(pulse.pulse.start * 1e-9, 9) - time)
                            time += round(pulse.pulse.duration * 1e-9, 9) + round(pulse.pulse.start * 1e-9, 9) - time
                            pulse.zhpulse.uid = pulse.zhpulse.uid + str(i)

                            weights_file = Path(
                                str(qibolab_folder)
                                + f"/runcards/{self.chip}/weights/integration_weights_optimization_qubit_{qubit.name}.npy"
                            )
                            if weights_file.is_file():
                                print("I'm using optimized IW")
                                # Optimal weights
                                samples = np.load(
                                    str(qibolab_folder)
                                    + f"/runcards/{self.chip}/weights/integration_weights_optimization_qubit_{qubit.name}.npy",
                                    allow_pickle=True,
                                )
                                if acquisition_type == lo.AcquisitionType.DISCRIMINATION:
                                    weight = lo.pulse_library.sampled_pulse_complex(
                                        uid="weight" + pulse.zhpulse.uid,
                                        samples=samples[0] * np.exp(1j * qubit.iq_angle),
                                    )
                                else:
                                    weight = lo.pulse_library.sampled_pulse_complex(
                                        uid="weight" + pulse.zhpulse.uid,
                                        samples=samples[0],
                                    )
                            else:
                                # Dumb weights
                                "We adjust for smearing and remove smearing/2 at the end"
                                exp.delay(signal=f"measure{qubit.name}", time=self.smearing)
                                if acquisition_type == lo.AcquisitionType.DISCRIMINATION:
                                    weight = lo.pulse_library.sampled_pulse_complex(
                                        np.ones([int(pulse.pulse.duration * 2)]) * np.exp(1j * qubit.iq_angle)
                                    )
                                else:
                                    weight = lo.pulse_library.const(
                                        uid="weight" + pulse.zhpulse.uid,
                                        length=round(pulse.pulse.duration * 1e-9, 9) - 1.5 * self.smearing,
                                        amplitude=1,
                                    )

                            exp.measure(
                                acquire_signal=f"acquire{qubit.name}",
                                handle=f"sequence{qubit.name}",
                                integration_kernel=weight,
                                integration_kernel_parameters=None,
                                integration_length=None,
                                measure_signal=f"measure{qubit.name}",
                                measure_pulse=pulse.zhpulse,
                                measure_pulse_length=round(pulse.pulse.duration * 1e-9, 9),
                                measure_pulse_parameters=None,
                                measure_pulse_amplitude=None,
                                acquire_delay=self.time_of_flight,
                                reset_delay=relaxation_time,
                            )
                            i += 1

    def fast_reset(self, exp, qubits, fast_reset):
        """fast reset after readout - small delay for signal processing"""
        print("Im fast resetting")
        for qubit_name in self.sequence_qibo.qubits:
            qubit = qubits[qubit_name]
            if not qubit.flux_coupler:
                with exp.section(uid=f"fast_reset{qubit.name}", play_after=f"sequence_measure{qubit.name}"):
                    # with exp.match_local(handle=f"acquire{qubit.name}"):
                    with exp.match_local(handle=f"sequence{qubit.name}"):
                        with exp.case(state=0):
                            pass
                        with exp.case(state=1):
                            exp.play(signal=f"drive{qubit.name}", pulse=ZhPulse(fast_reset[qubit.name]).zhpulse)

    def sweep(self, qubits, sequence, *sweepers, options):
        """Play pulse and sweepers sequence"""
        if options.relaxation_time is None:
            options.relaxation_time = self.relaxation_time

        # TODO: Read frequency for pulses instead of qubit patch
        for qubit in qubits.values():
            for pulse in sequence:
                if pulse.qubit == qubit.name:
                    if pulse.type is PulseType.READOUT:
                        qubit.readout_frequency = pulse.frequency
                    if pulse.type is PulseType.DRIVE:
                        qubit.drive_frequency = pulse.frequency
        """
        Play pulse sequence steps, one on each method:
        Translation, Calibration, Experiment Definition and Execution.
        """
        self.sweepers = list(sweepers)
        self.sequence_zh(sequence, qubits, sweepers)
        self.calibration_step(qubits)
        self.create_exp(qubits, options)
        self.run_exp()

        # TODO: General, several readouts and qubits
        # TODO: Implement the new results!
        "Get the results back"
        results = {}
        for qubit in qubits.values():
            if not qubit.flux_coupler:
                if self.sequence[f"readout{qubit.name}"]:
                    exp_res = self.results.get_data(f"sequence{qubit.name}")
                    print(exp_res)
                    i = exp_res.real
                    q = exp_res.imag
                    results[self.sequence[f"readout{qubit.name}"][0].pulse.serial] = ExecutionResults.from_components(
                        i, q
                    )

        # FIXME: Include this on the reports
        # html containing the pulse sequence schedule
        # lo.show_pulse_sheet("pulses", self.exp)

        # There is no reason for disconnection and it prevents reconnection
        # for a period of time making the software loops with execute_play_sequence crash
        # self.disconnect()
        return results

    # TODO: Recursion tests and Better sweeps logic
    def sweep_recursion(self, qubits, exp, exp_calib, relaxation_time, acquisition_type, fast_reset):
        """Sweepers recursion for multiple nested sweepers"""

        # Ordered like this for how they defined the frequncy sweep on qibocal to be the last
        # and I need it to be the outer one
        for sweep in self.sweepers:
            if sweep.parameter is Parameter.frequency:
                sweeper = sweep
                break
            sweeper = sweep

        i = len(self.sweepers) - 1
        self.sweepers.remove(sweeper)

        print(sweeper.parameter)
        parameter = None

        if sweeper.parameter is Parameter.frequency:
            for pulse in sweeper.pulses:
                qubit = pulse.qubit
                if pulse.type is PulseType.DRIVE:
                    line = "drive"
                elif pulse.type is PulseType.READOUT:
                    line = "measure"
                zhsweeper = ZhSweeper(pulse, sweeper, qubits[sweeper.pulses[0].qubit]).zhsweeper
                zhsweeper.uid = f"frequency"  # TODO: Changing the name from "frequency" breaks it
                exp_calib[f"{line}{qubit}"] = lo.SignalCalibration(
                    oscillator=lo.Oscillator(
                        frequency=zhsweeper,
                        modulation_type=lo.ModulationType.HARDWARE,
                    )
                )

        if sweeper.parameter is Parameter.bias:
            for qubit in sweeper.qubits.values():
                # for qubit in sweeper.qubits.values():
                #     qubit = sweeper.qubits[qubit]
                parameter = ZhSweeperLine(sweeper, qubit, self.sequence_qibo).zhsweeper

        elif sweeper.parameter is Parameter.delay:
            pulse = sweeper.pulses[0]
            qubit = pulse.qubit
            # for qubit in sweeper.qubits.values():
            #     qubit = sweeper.qubits[qubit]
            parameter = ZhSweeperLine(sweeper, qubit, self.sequence_qibo).zhsweeper

        elif parameter is None:
            parameter = ZhSweeper(sweeper.pulses[0], sweeper, qubits[sweeper.pulses[0].qubit]).zhsweeper

        with exp.sweep(
            uid=f"sweep_{sweeper.parameter.name.lower()}_{i}",  # FIXME: This uid fro double freq ???
            parameter=parameter,  # FIXME: This uid for double freq ???
            reset_oscillator_phase=True,
        ):
            if len(self.sweepers) > 0:
                self.sweep_recursion(qubits, exp, exp_calib, relaxation_time, acquisition_type, fast_reset)
            else:
                self.select_exp(exp, qubits, relaxation_time, acquisition_type, fast_reset)

    def sweep_recursion_NT(self, qubits, options, exp, exp_calib):
        """Sweepers recursion for multiple nested sweepers"""
        print("NT_loop")
        for sweep in self.NT_sweeps:
            sweeper = sweep

        i = len(self.NT_sweeps) - 1
        self.NT_sweeps.remove(sweeper)

        print(sweeper.parameter)
        parameter = None

        if sweeper.parameter is Parameter.amplitude:
            for pulse in sweeper.pulses:
                qubit = pulse.qubit
                if pulse.type is PulseType.DRIVE:
                    line = "drive"
                elif pulse.type is PulseType.READOUT:
                    line = "measure"
                zhsweeper = ZhSweeper(pulse, sweeper, qubits[sweeper.pulses[0].qubit]).zhsweeper
                zhsweeper.uid = f"amplitude"
                path = "DEV12146"

        # Leave it for dual freq if they dont work in RT
        if sweeper.parameter is Parameter.frequency:
            for pulse in sweeper.pulses:
                qubit = pulse.qubit
                if pulse.type is PulseType.DRIVE:
                    line = "drive"
                elif pulse.type is PulseType.READOUT:
                    line = "measure"
                zhsweeper = ZhSweeper(pulse, sweeper, qubits[sweeper.pulses[0].qubit]).zhsweeper
                zhsweeper.uid = f"frequency"  # TODO: Changing the name from "frequency" breaks it
                exp_calib[f"{line}{qubit}"] = lo.SignalCalibration(
                    oscillator=lo.Oscillator(
                        frequency=zhsweeper,
                        modulation_type=lo.ModulationType.HARDWARE,
                    )
                )

        if sweeper.parameter is Parameter.bias:
            for qubit in sweeper.qubits.values():
                # for qubit in sweeper.qubits.values():
                #     qubit = sweeper.qubits[qubit]
                parameter = ZhSweeperLine(sweeper, qubit, self.sequence_qibo).zhsweeper

        elif parameter is None:
            parameter = ZhSweeper(sweeper.pulses[0], sweeper, qubits[sweeper.pulses[0].qubit]).zhsweeper

        with exp.sweep(
            uid=f"sweep_{sweeper.parameter.name.lower()}_{i}",
            parameter=parameter,
        ):
            exp.set_node(
                path=f"/{path}/qachannels/*/oscs/0/gain",  # FIXME: Hardcoded SHFQA device
                value=parameter,
            )

            if len(self.NT_sweeps) > 0:
                self.sweep_recursion_NT(qubits, options, exp, exp_calib)
            else:
                self.define_exp(qubits, options, exp, exp_calib)

    # -----------------------------------------------------------------------------

    def play_sim(self, qubits, sequence, options, sim_time):
        """Play pulse sequence"""
        if options.relaxation_time is None:
            options.relaxation_time = self.relaxation_time

        self.sequence_zh(sequence, qubits, sweepers=[])
        self.calibration_step(qubits)
        self.create_exp(qubits, options)
        self.run_sim(sim_time)

    # TODO: Implement further pulse viewing functions from 2.2.0
    # should this be added in a way so the user can check how the sequence looks like ?
    def run_sim(self, sim_time):
        self.device_setup = lo.DeviceSetup.from_descriptor(
            yaml_text=self.descriptor,
            server_host="localhost",
            server_port="8004",
            setup_name=self.name,
        )
        # create a session
        self.sim_session = lo.Session(self.device_setup)
        # connect to session
        self.sim_device = self.session.connect(do_emulation=True)

        # Plot simulated output signals with helper function
        plot_simulation(
            self.exp,
            start_time=0,
            length=sim_time,
            plot_width=10,
            plot_height=3,
        )


# Try to convert this to plotty
def plot_simulation(
    compiled_experiment,
    start_time=0.0,
    length=2e-6,
    xaxis_label="Time (s)",
    yaxis_label="Amplitude",
    plot_width=6,
    plot_height=2,
):
    import re

    # additional imports for plotting
    import matplotlib.pyplot as plt
    import matplotlib.style as style

    # numpy for mathematics
    import numpy as np
    from matplotlib import cycler

    style.use("default")

    plt.rcParams.update(
        {
            "font.weight": "light",
            "axes.labelweight": "light",
            "axes.titleweight": "normal",
            "axes.prop_cycle": cycler(color=["#006699", "#FF0000", "#66CC33", "#CC3399"]),
            "svg.fonttype": "none",  # Make text editable in SVG
            "text.usetex": False,
        }
    )

    simulation = lo.OutputSimulator(compiled_experiment)

    mapped_signals = compiled_experiment.experiment.signal_mapping_status["mapped_signals"]

    xs = []
    y1s = []
    labels1 = []
    y2s = []
    labels2 = []
    for signal in mapped_signals:
        mapped_path = compiled_experiment.experiment.signals[signal].mapped_logical_signal_path

        full_path = re.sub(r"/logical_signal_groups/", "", mapped_path)
        signal_group_name = re.sub(r"/[^/]*$", "", full_path)
        signal_line_name = re.sub(r".*/", "", full_path)

        physical_channel_path = (
            compiled_experiment.device_setup.logical_signal_groups[signal_group_name]
            .logical_signals[signal_line_name]
            .physical_channel
        )

        my_snippet = simulation.get_snippet(
            compiled_experiment.device_setup.logical_signal_groups[signal_group_name]
            .logical_signals[signal_line_name]
            .physical_channel,
            start=start_time,
            output_length=length,
            get_trigger=True,
            get_frequency=True,
        )

        if "iq_channel" in str(physical_channel_path.type).lower() and "input" not in str(physical_channel_path.name):
            try:
                if my_snippet.time is not None:
                    xs.append(my_snippet.time)

                    y1s.append(my_snippet.wave.real)
                    labels1.append(f"{signal} I")

                    y2s.append(my_snippet.wave.imag)
                    labels2.append(f"{signal} Q")
            except Exception:
                pass

        if "iq_channel" not in str(physical_channel_path.type).lower() or "input" in physical_channel_path.name:
            try:
                if my_snippet.time is not None:
                    time_length = len(my_snippet.time)

                    xs.append(my_snippet.time)

                    y1s.append(my_snippet.wave.real)
                    labels1.append(f"{signal}")

                    empty_array = np.empty((1, time_length))
                    empty_array.fill(np.nan)
                    y2s.append(empty_array[0])
                    labels2.append(None)

            except Exception:
                pass

    colors = plt.rcParams["axes.prop_cycle"]()

    fig, axes = plt.subplots(
        nrows=len(y1s),
        sharex=False,
        figsize=(plot_width, len(mapped_signals) * plot_height),
    )

    if len(mapped_signals) > 1:
        for axs, x, y1, y2, label1, label2 in zip(axes.flat, xs, y1s, y2s, labels1, labels2):
            # Get the next color from the cycler
            c = next(colors)["color"]
            axs.plot(x, y1, label=label1, color=c)
            c = next(colors)["color"]
            axs.plot(x, y2, label=label2, color=c)
            axs.set_ylabel(yaxis_label)
            axs.set_xlabel(xaxis_label)
            axs.legend(loc="upper right")
            axs.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            axs.grid(True)

    elif len(mapped_signals) == 1:
        for x, y1, y2, label1, label2 in zip(xs, y1s, y2s, labels1, labels2):
            # Get the next color from the cycler
            c = next(colors)["color"]
            axes.plot(x, y1, label=label1, color=c)
            c = next(colors)["color"]
            axes.plot(x, y2, label=label2, color=c)
            axes.set_ylabel(yaxis_label)
            axes.set_xlabel(xaxis_label)
            axes.legend(loc="upper right")
            axes.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            axes.grid(True)

    fig.tight_layout()
    # fig.legend(loc="upper left")
    plt.savefig("Prueba.png")
    plt.show()
