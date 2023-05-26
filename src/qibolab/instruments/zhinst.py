import logging
import os
import warnings
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import laboneq._token
import laboneq.simple as lo
import numpy as np
from laboneq.contrib.example_helpers.plotting.plot_helpers import plot_simulation

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.paths import qibolab_folder
from qibolab.pulses import FluxPulse, PulseSequence, PulseType
from qibolab.sweeper import Parameter

# this env var just needs to be set
os.environ["LABONEQ_TOKEN"] = "not required"
laboneq._token.is_valid_token = lambda _token: True

# FIXME: Multiplex (For readout). Workaround integration weights padding with zeros.
# FIXME: Handle on acquires for list of pulse sequences
# FIXME: I think is a hardware limitation but I cant sweep multiple drive oscillator at the same time

NANO_TO_SECONDS = 1e-9
SERVER_PORT = "8004"
COMPILER_SETTINGS = {
    "SHFSG_FORCE_COMMAND_TABLE": True,
    "SHFSG_MIN_PLAYWAVE_HINT": 32,
    "SHFSG_MIN_PLAYZERO_HINT": 32,
}

"""Translating to Zurich ExecutionParameters"""
ACQUISITION_TYPE = {
    AcquisitionType.INTEGRATION: lo.AcquisitionType.INTEGRATION,
    AcquisitionType.RAW: lo.AcquisitionType.RAW,
    AcquisitionType.DISCRIMINATION: lo.AcquisitionType.DISCRIMINATION,
}

AVERAGING_MODE = {
    AveragingMode.CYCLIC: lo.AveragingMode.CYCLIC,
    AveragingMode.SINGLESHOT: lo.AveragingMode.SINGLE_SHOT,
}


# FIXME: Either implement more or create and arbitrary one
def select_pulse(pulse, pulse_type):
    """Pulse translation"""

    if str(pulse.shape) == "Rectangular()":
        return lo.pulse_library.const(
            uid=(f"{pulse_type}_{pulse.qubit}_"),
            length=round(pulse.duration * NANO_TO_SECONDS, 9),
            amplitude=pulse.amplitude,
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
        return lo.pulse_library.gaussian_square(
            uid=(f"{pulse_type}_{pulse.qubit}_"),
            length=round(pulse.duration * NANO_TO_SECONDS, 9),
            width=round(pulse.duration * NANO_TO_SECONDS, 9) * 0.9,  # 90% Flat
            amplitude=pulse.amplitude,
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

        # TODO: if "Slepian" in str(pulse.shape):
        "Implement Slepian shaped flux pulse https://arxiv.org/pdf/0909.5368.pdf"

        # TODO: if "Slepian" in str(pulse.shape):if "Sampled" in str(pulse.shape):
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


class ZhPulse:
    def __init__(self, pulse):
        """Zurich pulse from qibolab pulse"""
        self.pulse = pulse
        """Qibolab pulse"""
        self.signal = f"{pulse.type.name.lower()}{pulse.qubit}"
        """Line associated with the pulse"""
        self.zhpulse = select_pulse(pulse, pulse.type.name.lower())
        """Zurich pulse"""


class ZhSweeper:
    def __init__(self, pulse, sweeper, qubit):
        """
        Zurich sweeper from qibolab sweeper for pulse parameters
        Amplitude, Duration, Frequency (and maybe Phase)

        """

        self.sweeper = sweeper
        """Qibolab sweeper"""

        self.pulse = pulse
        """Qibolab pulse associated to the sweeper"""
        self.signal = f"{pulse.type.name.lower()}{pulse.qubit}"
        """Line associated with the pulse"""
        self.zhpulse = ZhPulse(pulse).zhpulse
        """Zurich pulse associated to the sweeper"""

        self.zhsweeper = self.select_sweeper(pulse.type, sweeper, qubit)
        """Zurich sweeper"""

        self.zhsweepers = [self.select_sweeper(pulse.type, sweeper, qubit)]
        """
        Zurich sweepers, Need something better to store multiple sweeps on the same pulse

        Not properly implemented as it was only used on Rabi amplitude vs lenght and it
        was an unused routine.
        """

    @staticmethod
    def select_sweeper(ptype, sweeper, qubit):
        """Sweeper translation"""

        if sweeper.parameter is Parameter.amplitude:
            return lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values,
            )
        if sweeper.parameter is Parameter.duration:
            return lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values * NANO_TO_SECONDS,
            )
        # TODO: take intermediate frequency from the pulses
        if sweeper.parameter is Parameter.frequency:
            if ptype is PulseType.READOUT:
                intermediate_frequency = qubit.readout_frequency - qubit.readout.local_oscillator.frequency
            elif ptype is PulseType.DRIVE:
                intermediate_frequency = qubit.drive_frequency - qubit.drive.local_oscillator.frequency
            return lo.LinearSweepParameter(
                uid=sweeper.parameter.name,
                start=sweeper.values[0] + intermediate_frequency,
                stop=sweeper.values[-1] + intermediate_frequency,
                count=len(sweeper.values),
            )

    def add_sweeper(self, sweeper, qubit):
        """Add sweeper to list of sweepers"""
        self.zhsweepers.append(self.select_sweeper(self.pulse.type, sweeper, qubit))


class ZhSweeperLine:
    def __init__(self, sweeper, qubit=None, sequence=None):
        """
        Zurich sweeper from qibolab sweeper for non pulse parameters
        Bias, Delay (, power_range, local_oscillator frequency, offset ???)

        For now Parameter.bias sweepers are implemented as Parameter.Amplitude
        on a flux pulse. We may want to keep this class separate for future
        Near Time sweeps
        """

        self.sweeper = sweeper
        """Qibolab sweeper"""

        # TODO: I already created a flux pulse check
        if sweeper.parameter is Parameter.bias:
            pulse = FluxPulse(
                start=sequence.start,
                duration=sequence.duration,
                amplitude=1,
                shape="Rectangular",
                channel=qubit.flux.name,
                qubit=qubit.name,
            )
            self.pulse = pulse
            self.signal = f"flux{qubit.name}"

            self.zhpulse = lo.pulse_library.const(
                uid=(f"{pulse.type.name.lower()}_{pulse.qubit}_"),
                length=round(pulse.duration * NANO_TO_SECONDS, 9),
                amplitude=pulse.amplitude,
            )

        # Need something better to store multiple sweeps on the same pulse
        self.zhsweeper = self.select_sweeper(sweeper)

    @staticmethod
    def select_sweeper(sweeper):
        """Sweeper translation"""
        if sweeper.parameter is Parameter.bias:
            return lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values,
            )
        if sweeper.parameter is Parameter.delay:
            return lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values * NANO_TO_SECONDS,
            )


class Zurich(AbstractInstrument):
    """Zurich driver main class"""

    def __init__(self, name, descriptor, use_emulation=False):
        self.name = name
        "Setup name (str)"

        self.descriptor = descriptor
        """
        Port and device mapping in yaml text (str)

        It should be used as a template by adding extra lines for each of the different
        frequency pulses played thought the same port after parsing the sequence.
        """

        self.emulation = use_emulation
        "Enable emulation mode (bool)"
        self.is_connected = False
        "Is the device connected ? (bool)"

        self.signal_map = {}
        "Signals to lines mapping"
        self.calibration = lo.Calibration()
        "Zurich calibration object)"

        self.device_setup = None
        self.session = None
        self.device = None
        "Zurich device parameters for connection"

        self.time_of_flight = 0.0
        self.smearing = 0.0
        self.chip = "iqm5q"
        "Parameters read from the runcard not part of ExecutionParameters"

        self.exp = None
        self.experiment = None
        self.exp_options = ExecutionParameters
        self.exp_calib = lo.Calibration()
        self.results = None
        "Zurich experiment definitions"

        self.acquisition_type = None
        "To store if the AcquisitionType.SPECTROSCOPY needs to be enabled by parsing the sequence"

        self.sequence = defaultdict(list)
        "Zurich pulse sequence"
        self.sequence_qibo = None
        # Remove if able

        self.sweepers = []
        self.nt_sweeps = None
        "Storing sweepers"
        # Improve the storing of multiple sweeps

    def connect(self):
        if not self.is_connected:
            for _ in range(3):
                try:
                    self.device_setup = lo.DeviceSetup.from_descriptor(
                        yaml_text=self.descriptor,
                        server_host="localhost",
                        server_port=SERVER_PORT,
                        setup_name=self.name,
                    )
                    # To fully remove logging #configure_logging=False
                    self.session = lo.Session(self.device_setup, log_level=30)
                    self.device = self.session.connect(do_emulation=self.emulation)
                    self.is_connected = True
                    break
                except Exception as exc:
                    logging.critical(f"Unable to connect:\n{str(exc)}\nRetrying...")
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
            logging.warning("Already disconnected")

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
                if len(self.sequence[f"drive{qubit.name}"]) != 0:
                    self.register_drive_line(
                        qubit=qubit,
                        intermediate_frequency=qubit.drive_frequency - qubit.drive.local_oscillator.frequency,
                    )
                if len(self.sequence[f"readout{qubit.name}"]) != 0:
                    self.register_readout_line(
                        qubit=qubit,
                        intermediate_frequency=qubit.readout_frequency - qubit.readout.local_oscillator.frequency,
                    )
        self.device_setup.set_calibration(self.calibration)

    def register_readout_line(self, qubit, intermediate_frequency):
        """Registers qubit measure and acquire lines to calibration and signal map.

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
                uid="lo_shfqa",
                frequency=int(qubit.readout.local_oscillator.frequency),
            ),
            range=qubit.readout.power_range,
            port_delay=None,
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
                uid="lo_shfqa",
                frequency=int(qubit.readout.local_oscillator.frequency),
            ),
            range=qubit.feedback.power_range,
            port_delay=self.time_of_flight * NANO_TO_SECONDS,
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
                uid="lo_shfqc",
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
            range=qubit.flux.power_range,
            port_delay=None,
            delay_signal=0,
            voltage_offset=qubit.flux.bias,
        )

    def run_exp(self):
        """Compilation settings, compilation step, execution step and data retrival"""
        compiler_settings = COMPILER_SETTINGS

        self.exp = self.session.compile(self.experiment, compiler_settings=compiler_settings)
        self.results = self.session.run(self.exp)

    @staticmethod
    def frequency_from_pulses(qubits, sequence):
        for pulse in sequence:
            qubit = qubits[pulse.qubit]
            if pulse.type is PulseType.READOUT:
                qubit.readout_frequency = pulse.frequency
            if pulse.type is PulseType.DRIVE:
                qubit.drive_frequency = pulse.frequency

    def experiment_flow(self, qubits, sequence, options, sweepers=[]):
        self.sequence_zh(sequence, qubits, sweepers)
        self.calibration_step(qubits)
        self.create_exp(qubits, options)

    # TODO: Play taking a big sequence with several acquire steps
    def play(self, qubits, sequence, options):
        """Play pulse sequence"""
        self.signal_map = {}
        dimensions = []
        if options.averaging_mode is AveragingMode.SINGLESHOT:
            dimensions = [options.nshots]

        # TODO: Read frequency for pulses instead of qubit patch
        self.frequency_from_pulses(qubits, sequence)

        """
        Play pulse sequence steps, one on each method:
        Translation, Calibration, Experiment Definition and Execution.
        """

        self.experiment_flow(qubits, sequence, options)
        self.run_exp()

        # TODO: General, several readouts and qubits
        # TODO: Implement the new results!
        "Get the results back"
        results = {}
        for qubit in qubits.values():
            if qubit.flux_coupler:
                continue
            q = qubit.name
            if len(self.sequence[f"readout{q}"]) != 0:
                exp_res = self.results.get_data(f"sequence{q}")
                if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                    data = np.array([exp_res]) if options.averaging_mode is AveragingMode.CYCLIC else np.array(exp_res)
                    results[self.sequence[f"readout{q}"][0].pulse.serial] = options.results_type(data)
                    results[self.sequence[f"readout{q}"][0].pulse.qubit] = options.results_type(data)
                else:
                    results[self.sequence[f"readout{q}"][0].pulse.serial] = options.results_type(data=np.array(exp_res))
                    results[self.sequence[f"readout{q}"][0].pulse.qubit] = options.results_type(data=np.array(exp_res))

        exp_dimensions = list(np.array(exp_res).shape)
        if dimensions != exp_dimensions:
            logging.warn("dimensions {: d} , exp_dimensions {: d}".format(dimensions, exp_dimensions))
            warnings.warn("dimensions not properly ordered")

        # FIXME: Include this on the reports
        # html containing the pulse sequence schedule
        # lo.show_pulse_sheet("pulses", self.exp)
        return results

    def sequence_zh(self, sequence, qubits, sweepers):
        """Qibo sequence to Zurich sequence"""

        "Define and assign the sequence"
        zhsequence = defaultdict(list)
        self.sequence_qibo = sequence

        "Fill the sequences with pulses according to their lines in temporal order"
        # TODO: Check if they invert the order if this will still work
        for pulse in sequence:
            zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"].append(ZhPulse(pulse))

        "Mess that gets the sweeper and substitutes the pulse it sweeps in the right place"

        SWEEPER_SET = {"amplitude", "frequency", "duration", "relative_phase"}
        SWEEPER_BIAS = {"bias"}
        SWEEPER_DELAY = {"delay"}

        for sweeper in sweepers:
            if sweeper.parameter.name in SWEEPER_SET:
                for pulse in sweeper.pulses:
                    aux_list = zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"]
                    if sweeper.parameter is Parameter.frequency and pulse.type is PulseType.READOUT:
                        self.acquisition_type = lo.AcquisitionType.SPECTROSCOPY
                    if sweeper.parameter is Parameter.amplitude and pulse.type is PulseType.READOUT:
                        self.acquisition_type = lo.AcquisitionType.SPECTROSCOPY
                        self.nt_sweeps.append(sweeper)
                        self.sweepers.remove(sweeper)
                    for element in aux_list:
                        if pulse == element.pulse:
                            if isinstance(aux_list[aux_list.index(element)], ZhPulse):
                                aux_list[aux_list.index(element)] = ZhSweeper(pulse, sweeper, qubits[pulse.qubit])
                            elif isinstance(aux_list[aux_list.index(element)], ZhSweeper):
                                aux_list[aux_list.index(element)].add_sweeper(sweeper, qubits[pulse.qubit])

            if sweeper.parameter.name in SWEEPER_BIAS:
                for qubit in sweeper.qubits:
                    zhsequence[f"flux{qubit.name}"] = [ZhSweeperLine(sweeper, qubit, sequence)]

            # FIXME: This may not place the Zhsweeper when the delay occurs among different sections or lines
            if sweeper.parameter.name in SWEEPER_DELAY:
                pulse = sweeper.pulses[0]
                aux_list = zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"]
                for element in aux_list:
                    if pulse == element.pulse:
                        if isinstance(aux_list[aux_list.index(element)], ZhPulse):
                            aux_list.insert(
                                aux_list.index(element) + 1,
                                ZhSweeperLine(sweeper, pulse.qubit, sequence),
                            )
                            break

        self.sequence = zhsequence

    def create_exp(self, qubits, options):
        """Zurich experiment initialization usig their Experiment class"""

        """Setting experiment signal lines"""
        signals = []
        for qubit in qubits.values():
            q = qubit.name
            if qubit.flux_coupler:
                signals.append(lo.ExperimentSignal(f"flux{q}"))
            else:
                if len(self.sequence[f"drive{q}"]) != 0:
                    signals.append(lo.ExperimentSignal(f"drive{q}"))
                if qubit.flux is not None:
                    signals.append(lo.ExperimentSignal(f"flux{q}"))
                if len(self.sequence[f"readout{q}"]) != 0:
                    signals.append(lo.ExperimentSignal(f"measure{q}"))
                    signals.append(lo.ExperimentSignal(f"acquire{q}"))

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
        exp_options = replace(options, acquisition_type=acquisition_type, averaging_mode=averaging_mode)

        exp_calib = lo.Calibration()
        """Near Time recursion loop or directly to Real Time recursion loop"""
        if self.nt_sweeps is not None:
            self.sweep_recursion_nt(qubits, exp_options, exp, exp_calib)
        else:
            self.define_exp(qubits, exp_options, exp, exp_calib)

    def define_exp(self, qubits, exp_options, exp, exp_calib):
        """Real time definition"""
        with exp.acquire_loop_rt(
            uid="shots",
            count=exp_options.nshots,
            acquisition_type=exp_options.acquisition_type,
            averaging_mode=exp_options.averaging_mode,
        ):
            """Recursion loop for sweepers or just play a sequence"""
            if len(self.sweepers) > 0:
                self.sweep_recursion(qubits, exp, exp_calib, exp_options)
            else:
                self.select_exp(exp, qubits, exp_options)
            exp.set_calibration(exp_calib)
            exp.set_signal_map(self.signal_map)
            self.experiment = exp

    def select_exp(self, exp, qubits, exp_options):
        """Build Zurich Experiment selecting the relevant sections"""
        if "drive" in str(self.sequence):
            if "flux" in str(self.sequence):
                self.flux(exp, qubits)
                self.drive(exp, qubits)
            else:
                self.drive(exp, qubits)
        elif "flux" in str(self.sequence):
            self.flux(exp, qubits)
        self.measure_relax(exp, qubits, exp_options.relaxation_time, exp_options.acquisition_type)
        if exp_options.fast_reset is not False:
            self.fast_reset(exp, qubits, exp_options.fast_reset)

    @staticmethod
    def play_sweep_select(exp, qubit, pulse, section, parameters, partial_sweep):
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
        elif "frequency" in partial_sweep.uid or partial_sweep.uid == "delay":
            # see if below also works for consistency
            # elif any("frequency" in param for param in parameters) or any("delay" in param for param in parameters):
            exp.play(
                signal=f"{section}{qubit.name}",
                pulse=pulse.zhpulse,
                phase=pulse.pulse.relative_phase,
            )

    def play_sweep(self, exp, qubit, pulse, section):
        """Play Zurich pulse when a sweeper is involved"""

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
            self.play_sweep_select(exp, qubit, pulse, section, parameters, partial_sweep)

    def flux(self, exp, qubits):
        """qubit flux or qubit coupler flux for bias sweep or pulses"""
        for qubit in qubits.values():
            q = qubit.name
            with exp.section(uid=f"sequence_bias{q}"):
                i = 0
                time = 0
                for pulse in self.sequence[f"flux{q}"]:
                    if not isinstance(pulse, ZhSweeperLine):
                        pulse.zhpulse.uid += str(i)
                        exp.delay(
                            signal=f"flux{q}",
                            time=round(pulse.pulse.start * NANO_TO_SECONDS, 9) - time,
                        )
                        time = round(pulse.pulse.duration * NANO_TO_SECONDS, 9) + round(
                            pulse.pulse.start * NANO_TO_SECONDS, 9
                        )
                    if isinstance(pulse, ZhSweeperLine):
                        self.play_sweep(exp, qubit, pulse, section="flux")
                    else:
                        exp.play(signal=f"flux{q}", pulse=pulse.zhpulse)
                    i += 1

    def drive(self, exp, qubits):
        """qubit driving pulses"""
        for qubit in qubits.values():
            if qubit.flux_coupler:
                continue
            q = qubit.name
            time = 0
            i = 0
            if len(self.sequence[f"drive{q}"]) != 0:
                with exp.section(uid=f"sequence_drive{q}"):
                    for pulse in self.sequence[f"drive{q}"]:
                        if not isinstance(pulse, ZhSweeperLine):
                            exp.delay(
                                signal=f"drive{q}",
                                time=round(pulse.pulse.start * NANO_TO_SECONDS, 9) - time,
                            )
                            time = round(pulse.pulse.duration * NANO_TO_SECONDS, 9) + round(
                                pulse.pulse.start * NANO_TO_SECONDS, 9
                            )
                            pulse.zhpulse.uid += str(i)
                            if isinstance(pulse, ZhSweeper):
                                self.play_sweep(exp, qubit, pulse, section="drive")
                            elif isinstance(pulse, ZhPulse):
                                exp.play(
                                    signal=f"drive{q}",
                                    pulse=pulse.zhpulse,
                                    phase=pulse.pulse.relative_phase,
                                )
                                i += 1
                        elif isinstance(pulse, ZhSweeperLine):
                            exp.delay(signal=f"drive{q}", time=pulse.zhsweeper)

    @staticmethod
    def play_after_set(sequence, type):
        longest = 0
        for pulse in sequence:
            if longest < pulse.finish:
                longest = pulse.finish
                qubit_after = pulse.qubit
        return f"sequence_{type}{qubit_after}"

    # For pulsed spectroscopy, set integration_length and either measure_pulse or measure_pulse_length.
    # For CW spectroscopy, set only integration_length and do not specify the measure signal.
    # For all other measurements, set either length or pulse for both the measure pulse and integration kernel.
    def measure_relax(self, exp, qubits, relaxation_time, acquisition_type):
        """qubit readout pulse, data acquisition and qubit relaxation"""
        play_after = None
        if len(self.sequence_qibo.qf_pulses) != 0 and len(self.sequence_qibo.qd_pulses) != 0:
            play_after = (
                self.play_after_set(self.sequence_qibo.qf_pulses, "bias")
                if self.sequence_qibo.qf_pulses.finish > self.sequence_qibo.qd_pulses.finish
                else self.play_after_set(self.sequence_qibo.qd_pulses, "drive")
            )
        elif len(self.sequence_qibo.qf_pulses) != 0:
            play_after = self.play_after_set(self.sequence_qibo.qf_pulses, "bias")
        elif len(self.sequence_qibo.qd_pulses) != 0:
            play_after = self.play_after_set(self.sequence_qibo.qd_pulses, "drive")

        for qubit in qubits.values():
            if qubit.flux_coupler:
                continue
            q = qubit.name
            if len(self.sequence[f"readout{q}"]) != 0:
                for pulse in self.sequence[f"readout{q}"]:
                    i = 0
                    with exp.section(uid=f"sequence_measure{q}", play_after=play_after):
                        pulse.zhpulse.uid += str(i)

                        """Integration weights definition or load from the chip folder"""
                        weights_file = Path(
                            str(qibolab_folder)
                            + f"/runcards/{self.chip}/weights/integration_weights_optimization_qubit_{q}.npy"
                        )
                        if weights_file.is_file():
                            logging.info("I'm using optimized IW")
                            samples = np.load(
                                str(qibolab_folder)
                                + f"/runcards/{self.chip}/weights/integration_weights_optimization_qubit_{q}.npy",
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
                            logging.info("I'm using dumb IW")
                            "We adjust for smearing and remove smearing/2 at the end"
                            exp.delay(
                                signal=f"measure{q}",
                                time=self.smearing * NANO_TO_SECONDS,
                            )
                            if acquisition_type == lo.AcquisitionType.DISCRIMINATION:
                                weight = lo.pulse_library.sampled_pulse_complex(
                                    np.ones([int(pulse.pulse.duration * 2 - 3 * self.smearing * NANO_TO_SECONDS)])
                                    * np.exp(1j * qubit.iq_angle)
                                )
                            else:
                                weight = lo.pulse_library.const(
                                    uid="weight" + pulse.zhpulse.uid,
                                    length=round(pulse.pulse.duration * NANO_TO_SECONDS, 9)
                                    - 1.5 * self.smearing * NANO_TO_SECONDS,
                                    amplitude=1,
                                )

                        measure_pulse_parameters = {"phase": 0}

                        exp.measure(
                            acquire_signal=f"acquire{q}",
                            handle=f"sequence{q}",
                            integration_kernel=weight,
                            integration_kernel_parameters=None,
                            integration_length=None,
                            measure_signal=f"measure{q}",
                            measure_pulse=pulse.zhpulse,
                            measure_pulse_length=round(pulse.pulse.duration * NANO_TO_SECONDS, 9),
                            measure_pulse_parameters=measure_pulse_parameters,
                            measure_pulse_amplitude=None,
                            acquire_delay=self.time_of_flight * NANO_TO_SECONDS,
                            reset_delay=relaxation_time * NANO_TO_SECONDS,
                        )
                        i += 1

    def fast_reset(self, exp, qubits, fast_reset):
        """
        Conditional fast reset after readout - small delay for signal processing
        This is a very naive approach that can be improved by repeating this step until
        we reach non fast reset fidelity
        https://quantum-computing.ibm.com/lab/docs/iql/manage/systems/reset/backend_reset
        """
        logging.warning("Im fast resetting")
        for qubit_name in self.sequence_qibo.qubits:
            qubit = qubits[qubit_name]
            if qubit.flux_coupler:
                continue
            q = qubit.name
            with exp.section(uid=f"fast_reset{q}", play_after=f"sequence_measure{q}"):
                with exp.match_local(handle=f"sequence{q}"):
                    with exp.case(state=1):
                        exp.play(signal=f"drive{q}", pulse=ZhPulse(fast_reset[q]).zhpulse)

    def sweep(self, qubits, sequence: PulseSequence, options, *sweepers):
        """Play pulse and sweepers sequence"""

        self.signal_map = {}

        sweepers = list(sweepers)

        dimensions = []
        if options.averaging_mode is AveragingMode.SINGLESHOT:
            dimensions = [options.nshots]

        for sweeper in sweepers:
            dimensions.append(len(sweeper.values))

        # Re-arranging sweepers based on hardware limitations
        # FIXME: Punchout and frequency case
        rearranging_axes = [[], []]
        if len(sweepers) == 2:
            if sweepers[1].parameter is Parameter.frequency:
                if sweepers[0].parameter is Parameter.bias:
                    rearranging_axes[0] += [sweepers.index(sweepers[1])]
                    rearranging_axes[1] += [0]
                    sweeper_changed = sweepers[1]
                    sweepers.remove(sweeper_changed)
                    sweepers.insert(0, sweeper_changed)
                    warnings.warn("Sweepers were reordered")
                elif (
                    not sweepers[0].parameter is Parameter.amplitude
                    and sweepers[0].pulses.type is not PulseType.READOUT
                ):
                    rearranging_axes[0] += [sweepers.index(sweepers[1])]
                    rearranging_axes[1] += [0]
                    sweeper_changed = sweepers[1]
                    sweepers.remove(sweeper_changed)
                    sweepers.insert(0, sweeper_changed)
                    warnings.warn("Sweepers were reordered")

        # TODO: Read frequency for pulses instead of qubit patch
        self.frequency_from_pulses(qubits, sequence)

        """
        Play pulse sequence steps, one on each method:
        Translation, Calibration, Experiment Definition and Execution.
        """
        self.sweepers = sweepers

        self.experiment_flow(qubits, sequence, options, sweepers)
        self.run_exp()

        # TODO: General, several readouts and qubits
        # TODO: Implement the new results!
        "Get the results back"
        results = {}
        for qubit in qubits.values():
            if qubit.flux_coupler:
                continue
            q = qubit.name
            if len(self.sequence[f"readout{q}"]) != 0:
                exp_res = self.results.get_data(f"sequence{q}")
                # Reorder dimensions
                exp_res = np.moveaxis(exp_res, rearranging_axes[0], rearranging_axes[1])
                if options.acquisition_type is AcquisitionType.DISCRIMINATION:
                    data = np.array([exp_res]) if options.averaging_mode is AveragingMode.CYCLIC else np.array(exp_res)
                    results[self.sequence[f"readout{q}"][0].pulse.serial] = options.results_type(data)
                else:
                    results[self.sequence[f"readout{q}"][0].pulse.serial] = options.results_type(data=np.array(exp_res))

        exp_dimensions = list(np.array(exp_res).shape)
        if dimensions != exp_dimensions:
            logging.warn("dimensions {: d} , exp_dimensions {: d}".format(dimensions, exp_dimensions))
            warnings.warn("dimensions not properly ordered")

        for sigout in range(0, 8):
            self.session.devices["device_hdawg"].awgs[0].sigouts[sigout].offset = 0
        self.session.devices["device_hdawg2"].awgs[0].sigouts[0].offset = 0

        # FIXME: Include this on the reports
        # html containing the pulse sequence schedule
        # lo.show_pulse_sheet("pulses", self.exp)
        return results

    def sweep_recursion(self, qubits, exp, exp_calib, exp_options):
        """Sweepers recursion for multiple nested Real Time sweepers"""

        sweeper = self.sweepers[0]

        i = len(self.sweepers) - 1
        self.sweepers.remove(sweeper)
        parameter = None

        if sweeper.parameter is Parameter.frequency:
            for pulse in sweeper.pulses:
                line = "drive" if pulse.type is PulseType.DRIVE else "measure"
                zhsweeper = ZhSweeper(pulse, sweeper, qubits[sweeper.pulses[0].qubit]).zhsweeper
                zhsweeper.uid = f"frequency"  # TODO: Changing the name from "frequency" breaks it
                exp_calib[f"{line}{pulse.qubit}"] = lo.SignalCalibration(
                    oscillator=lo.Oscillator(
                        frequency=zhsweeper,
                        modulation_type=lo.ModulationType.HARDWARE,
                    )
                )
        if sweeper.parameter is Parameter.amplitude:
            for pulse in sweeper.pulses:
                pulse.amplitude *= max(sweeper.values)
                sweeper.values /= max(sweeper.values)
                parameter = ZhSweeper(pulse, sweeper, qubits[sweeper.pulses[0].qubit]).zhsweeper

        if sweeper.parameter is Parameter.bias:
            for qubit in sweeper.qubits:
                parameter = ZhSweeperLine(sweeper, qubit, self.sequence_qibo).zhsweeper

        elif sweeper.parameter is Parameter.delay:
            parameter = ZhSweeperLine(sweeper).zhsweeper

        elif parameter is None:
            parameter = ZhSweeper(sweeper.pulses[0], sweeper, qubits[sweeper.pulses[0].qubit]).zhsweeper

        with exp.sweep(
            uid=f"sweep_{sweeper.parameter.name.lower()}_{i}",  # FIXME: This uid trouble double freq ???
            parameter=parameter,  # FIXME: This uid for double freq ???
            reset_oscillator_phase=True,  # FIXME: Should we reset this phase ???
        ):
            if len(self.sweepers) > 0:
                self.sweep_recursion(qubits, exp, exp_calib, exp_options)
            else:
                self.select_exp(exp, qubits, exp_options)

    def sweep_recursion_nt(self, qubits, options, exp, exp_calib):
        """
        Sweepers recursion for Near Time sweepers. Faster than regular software sweepers as
        they are executed on the actual device by (software ? or slower hardware ones)

        You want to avoid them so for now they are implement for a specific sweep.
        """

        logging.info("nt Loop")

        sweeper = self.nt_sweeps[0]

        i = len(self.nt_sweeps) - 1
        self.nt_sweeps.remove(sweeper)

        parameter = None

        if sweeper.parameter is Parameter.amplitude:
            for pulse in sweeper.pulses:
                zhsweeper = ZhSweeper(pulse, sweeper, qubits[sweeper.pulses[0].qubit]).zhsweeper
                zhsweeper.uid = "amplitude"  # f"amplitude{i}"
                path = "DEV12146"  # Hardcoded for SHFQC(SHFQA)
                parameter = zhsweeper

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

            if len(self.nt_sweeps) > 0:
                self.sweep_recursion_nt(qubits, options, exp, exp_calib)
            else:
                self.define_exp(qubits, options, exp, exp_calib)

    # -----------------------------------------------------------------------------

    def play_sim(self, qubits, sequence, options, sim_time):
        """Play pulse sequence"""

        self.experiment_flow(sequence, qubits, options)
        self.exp = self.session.compile(self.experiment)
        self.run_sim(sim_time)

    # TODO: Implement further pulse viewing functions from 2.2.0
    # should this be added in a way so the user can check how the sequence looks like ?
    def run_sim(self, sim_time):
        self.device_setup = lo.DeviceSetup.from_descriptor(
            yaml_text=self.descriptor,
            server_host="localhost",
            server_port=SERVER_PORT,
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
            xaxis_label="Time (s)",
            yaxis_label="Amplitude",
            plot_width=10,
            plot_height=3,
        )
