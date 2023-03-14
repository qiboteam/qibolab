from collections import defaultdict

import laboneq.simple as lo
import numpy as np

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.pulses import FluxPulse, Pulse
from qibolab.result import ExecutionResults

# TODO: Simulation
# session = Session(device_setup=device_setup)
# session.connect(do_emulation=use_emulation)
# my_results = session.run(exp)

# FIXME: Multiplex (For readout)
# FIXME: Lenght on pulses
# FIXME: Handle on acquires
# FIXME: I think is a hardware limitation but I cant sweep multiple drive oscillator at the same time
# FIXME: Docs & tests

###TEST
# TODO: Fast Reset
# TODO: Loops for multiple qubits [Parallel and Nested]
# TODO:For play Single shot
# TODO:For play Discrimination


class ZhPulse:
    def __init__(self, pulse):
        self.pulse = pulse
        self.signal = f"{pulse.type.name.lower()}{pulse.qubit}"
        self.zhpulse = self.select_pulse(pulse, pulse.type.name.lower())

        # For discrimination ?
        self.shot = None
        self.IQ = None
        self.shots = None
        self.threshold = None

    # Either implement more or create and arbitrary one
    def select_pulse(self, pulse, type):
        if str(pulse.shape) == "Rectangular()":
            Zh_Pulse = lo.pulse_library.const(
                uid=(f"{type}_{pulse.qubit}_"),
                length=round(pulse.duration * 1e-9, 9),
                amplitude=pulse.amplitude,
            )
        elif "Gaussian" in str(pulse.shape):
            sigma = pulse.shape.rel_sigma
            Zh_Pulse = lo.pulse_library.gaussian(
                uid=(f"{type}_{pulse.qubit}_"),
                length=round(pulse.duration * 1e-9, 9),
                amplitude=pulse.amplitude,
                sigma=2 / sigma,
            )
        elif "Drag" in str(pulse.shape):
            sigma = pulse.shape.rel_sigma
            beta = pulse.shape.beta
            Zh_Pulse = lo.pulse_library.drag(
                uid=(f"{type}_{pulse.qubit}_"),
                length=round(pulse.duration * 1e-9, 9),
                amplitude=pulse.amplitude,
                sigma=2 / sigma,
                beta=beta,
            )
        return Zh_Pulse


# for pulse sweeps
class ZhSweeper:
    def __init__(self, pulse, sweeper, qubit):
        self.sweeper = sweeper

        self.pulse = pulse
        self.signal = f"{pulse.type.name.lower()}{pulse.qubit}"
        self.zhpulse = ZhPulse(pulse).zhpulse

        # Need something better to store multiple sweeps on the same pulse
        self.zhsweeper = self.select_sweeper(sweeper, sweeper.parameter.name, qubit)
        self.zhsweepers = [self.select_sweeper(sweeper, sweeper.parameter.name, qubit)]

        # Stores the baking object (for pulses that need 1ns resolution)
        self.shot = None
        self.IQ = None
        self.threshold = None

    # Does LinearSweepParameter vs SweepParameter provide any advantage ?
    def select_sweeper(self, sweeper, parameter, qubit):
        # TODO: Join when finished(amplitude, duration, delay)
        if parameter == "amplitude":
            Zh_Sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values,
            )
        elif parameter == "duration":
            Zh_Sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values * 1e-9,
            )
        elif parameter == "frequency":
            if self.pulse.type.name.lower() == "readout":
                intermediate_frequency = qubit.readout_frequency - qubit.readout.local_oscillator.frequency
            elif self.pulse.type.name.lower() == "drive":
                intermediate_frequency = qubit.drive_frequency - qubit.drive.local_oscillator.frequency
            Zh_Sweeper = lo.LinearSweepParameter(
                uid=sweeper.parameter.name,
                start=sweeper.values[0] + intermediate_frequency,
                stop=sweeper.values[-1] + intermediate_frequency,
                count=len(sweeper.values),
            )

        return Zh_Sweeper

    def add_sweeper(self, sweeper, qubit):
        self.zhsweepers.append(self.select_sweeper(sweeper, sweeper.parameter.name, qubit))


# for no pulse sweeps
class ZhSweeper_line:
    def __init__(self, sweeper, qubit, sequence):
        self.sweeper = sweeper

        if sweeper.parameter.name == "bias":
            self.signal = f"flux{qubit.name}"
            self.zhpulse = lo.pulse_library.const(
                uid=(f"flux_{qubit.name}_"),
                length=round(sequence.duration * 1e-9, 9),  # 3.0e-6,
            )

        # Need something better to store multiple sweeps on the same pulse
        self.zhsweeper = self.select_sweeper(sweeper, sweeper.parameter.name)

    # Does LinearSweepParameter vs SweepParameter provide any advantage ?
    def select_sweeper(self, sweeper, parameter):
        if parameter == "bias":
            Zh_Sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values,
            )
        elif parameter == "delay":
            Zh_Sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values * 1e-9,
            )

        return Zh_Sweeper


class Zurich(AbstractInstrument):
    def __init__(self, name, descriptor, use_emulation=False):
        self.name = name
        self.descriptor = descriptor
        self.emulation = use_emulation

        self.is_connected = False

        self.signal_map = {}
        self.calibration = lo.Calibration()

        self.device_setup = None
        self.session = None
        self.device = None

        self.relaxation_time = 0.0
        self.time_of_flight = 0.0

        self.exp = None
        self.experiment = None
        self.results = None

        self.sequence = None
        self.acquisition_type = None

        # Improve the storing of multiple sweeps
        self.nsweeps = 0.0
        self.sweeps = None
        self.sweepers = None

        # Remove if able
        self.sequence_qibo = None

    def connect(self):
        if not self.is_connected:
            for attempt in range(3):
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
            print(f"Already disconnected")

    def setup(self, qubits, relaxation_time=0, time_of_flight=0, smearing=0, **_kwargs):
        self.relaxation_time = relaxation_time
        self.time_of_flight = time_of_flight

    def calibration_step(self, qubits):
        self.signal_map = {}
        self.calibration = lo.Calibration()

        # Map can be done in experiment, but general calibration needs to be here
        # General calibration != experiment calibration
        for qubit in qubits.values():
            if "c" in str(qubit.name):
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
                        Soft_Mod=True,
                    )
        self.device_setup.set_calibration(self.calibration)

    def register_readout_line(self, qubit, intermediate_frequency, Soft_Mod=True):
        """Registers qubit measure and acquire lines to calibration and signal map."""

        if Soft_Mod:
            modulation_type = lo.ModulationType.SOFTWARE
        else:
            modulation_type = lo.ModulationType.HARDWARE

        q = qubit.name
        self.signal_map[f"measure{q}"] = self.device_setup.logical_signal_groups[f"q{q}"].logical_signals[
            "measure_line"
        ]
        self.calibration[f"/logical_signal_groups/q{q}/measure_line"] = lo.SignalCalibration(
            oscillator=lo.Oscillator(
                frequency=intermediate_frequency,
                modulation_type=modulation_type,
            ),
            local_oscillator=lo.Oscillator(
                frequency=int(qubit.readout.local_oscillator.frequency),
                # frequency=0.0, # This and PortMode.LF allow debugging with and oscilloscope
            ),
            range=qubit.readout.power_range,
            port_delay=None,
            # port_mode= lo.PortMode.LF,
            delay_signal=0,  # self.time_of_flight,
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
            port_delay=150e-9,  # self.time_of_flight,
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
            range=qubit.flux.power_range, port_delay=None, delay_signal=0, voltage_offset=qubit.flux.offset
        )

    def run_exp(self):
        # Compiler settings required for active reset and multiplex.
        compiler_settings = {
            "SHFSG_FORCE_COMMAND_TABLE": True,
            "SHFSG_MIN_PLAYWAVE_HINT": 32,
            "SHFSG_MIN_PLAYZERO_HINT": 32,
        }

        self.exp = self.session.compile(self.experiment, compiler_settings=compiler_settings)
        self.results = self.session.run(self.exp)

    def play(self, qubits, sequence, nshots, relaxation_time, fast_reset, sim_time=10e-6):
        if relaxation_time is None:
            relaxation_time = self.relaxation_time

        self.sequence_zh(sequence, qubits, sweepers=[])
        self.calibration_step(qubits)
        self.create_exp(qubits, nshots, relaxation_time, fast_reset)
        self.run_exp()
        self.disconnect()

        # TODO: General, several readouts and qubits
        results = {}
        for qubit in qubits.values():
            if "c" in str(qubit.name):
                pass
            else:
                if self.sequence[f"readout{qubit.name}"]:
                    exp_res = self.results.get_data(f"sequence{qubit.name}")
                    i = exp_res.real
                    q = exp_res.imag
                    results[self.sequence[f"readout{qubit.name}"][0].pulse.serial] = ExecutionResults.from_components(
                        i, q
                    )
        self.run_sim(sim_time)
        # lo.show_pulse_sheet("pulses", self.exp) #FIXME: Wait until next LO release for bugs
        return results

    # TODO: Store the sweepers nice[Find a way to store nested vs parallel sweeps]
    def sequence_zh(self, sequence, qubits, sweepers):
        Zhsequence = defaultdict(list)
        self.sequence_qibo = sequence
        for qubit in qubits.values():
            pulse = FluxPulse(
                start=0,
                duration=sequence.duration,
                amplitude=qubit.sweetspot,
                shape="Rectangular",
                channel=qubit.flux.name,
                qubit=qubit.name,
            )
            Zhsequence[f"flux{pulse.qubit}"].append(ZhPulse(pulse))

        for pulse in sequence:
            Zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"].append(ZhPulse(pulse))

        # For single qubit sweeps for now with one pulse
        sweeps_all = {}
        sweeps = []
        nsweeps = 0
        for sweeper in sweepers:
            nsweeps += 1
            if (
                sweeper.parameter.name == "amplitude"
                or sweeper.parameter.name == "frequency"
                or sweeper.parameter.name == "duration"
                or sweeper.parameter.name == "relative_phase"
            ):
                for pulse in sweeper.pulses:
                    aux_list = Zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"]
                    if sweeper.parameter.name == "frequency" and pulse.type.name.lower() == "readout":
                        self.acquisition_type = lo.AcquisitionType.SPECTROSCOPY
                    for element in aux_list:
                        if pulse == element.pulse:
                            if isinstance(aux_list[aux_list.index(element)], ZhPulse):
                                aux_list[aux_list.index(element)] = ZhSweeper(pulse, sweeper, qubits[pulse.qubit])
                            elif isinstance(aux_list[aux_list.index(element)], ZhSweeper):
                                aux_list[aux_list.index(element)].add_sweeper(sweeper, qubits[pulse.qubit])
                            sweeps.append(ZhSweeper(pulse, sweeper, qubits[pulse.qubit]))
            elif sweeper.parameter.name == "bias":
                for qubit in sweeper.qubits.values():
                    sweeps.append(ZhSweeper_line(sweeper, qubit, sequence))
                    Zhsequence[f"flux{qubit.name}"] = [ZhSweeper_line(sweeper, qubit, sequence)]

            # FIXME: This may not place the Zhsweeper when the delay occurs among different sections
            elif sweeper.parameter.name == "delay":
                pulse = sweeper.pulses[0]
                qubit = pulse.qubit
                aux_list = Zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"]
                for element in aux_list:
                    if isinstance(element, ZhPulse):
                        if pulse == element.pulse:
                            if isinstance(aux_list[aux_list.index(element)], ZhPulse):
                                aux_list.insert(aux_list.index(element) + 1, ZhSweeper_line(sweeper, qubit, sequence))
                                sweeps.append(ZhSweeper_line(sweeper, qubit, sequence))

        sweeps_all[nsweeps] = sweeps
        self.sequence = Zhsequence
        self.nsweeps = nsweeps  # Should count the dimension of the sweep 1D, 2D etc
        self.sweeps = sweeps_all

    def create_exp(self, qubits, nshots, relaxation_time, fast_reset, **Params):
        signals = []
        for qubit in qubits.values():
            if "c" in str(qubit.name):
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

        if self.acquisition_type is None:
            self.acquisition_type = lo.AcquisitionType.INTEGRATION

        with exp.acquire_loop_rt(
            uid="shots",
            count=nshots,
            # repetition_mode= lo.RepetitionMode.CONSTANT,
            # repetition_time= 50e-6,
            acquisition_type=self.acquisition_type,
            # acquisition_type=lo.AcquisitionType.DISCRIMINATION,
            averaging_mode=lo.AveragingMode.CYCLIC,
            # averaging_mode=lo.AveragingMode.SINGLE_SHOT,
        ):
            if self.nsweeps > 0:
                exp_calib = lo.Calibration()
                self.sweep_recursion(qubits, exp, exp_calib, relaxation_time)
                exp.set_calibration(exp_calib)
            else:
                self.select_exp(exp, qubits, relaxation_time, fast_reset)
            self.experiment = exp
            exp.set_signal_map(self.signal_map)

    def select_exp(self, exp, qubits, relaxation_time, fast_reset):
        if "drive" in str(self.sequence):
            if "flux" in str(self.sequence):
                self.Flux(exp, qubits)
                self.Drive(exp, qubits)
            else:
                self.Drive(exp, qubits)
        elif "flux" in str(self.sequence):
            self.Flux(exp, qubits)
        self.Measure(exp, qubits)
        self.qubit_reset(exp, qubits, relaxation_time, fast_reset)

    # FIXME: This seems to create the 2D issues ?!?
    def play_sweep(self, exp, qubit, pulse, section):
        # FIXME: This loop for when a pulse is swept with several parameters(Max:3[Lenght, Amplitude, Phase])
        if len(self.sweeps) == 100:  # Need a better way of checking
            exp.play(
                signal=f"{section}{qubit.name}",
                pulse=pulse.zhpulse,
                amplitude=pulse.zhsweepers[0],
                length=pulse.zhsweepers[1],
                phase=pulse.pulse.relative_phase,
            )

        # This loop for when a pulse is swept with one or none parameters ???
        elif isinstance(pulse, ZhSweeper_line):
            parameters = pulse.zhsweeper.uid
            print("parameters", parameters)
            if parameters == "bias":
                # if any("bias" in param for param in parameters):
                exp.play(
                    signal=f"{section}{qubit.name}",
                    pulse=pulse.zhpulse,
                    amplitude=pulse.zhsweeper,
                )
                print("pulse", pulse.zhpulse)

        else:
            parameters = []
            for partial_sweep in pulse.zhsweepers:
                parameters.append(partial_sweep.uid)
            print("parameters", parameters)
            if any("amplitude" in param for param in parameters):
                exp.play(
                    signal=f"{section}{qubit.name}",
                    pulse=pulse.zhpulse,
                    amplitude=pulse.zhsweeper,
                    phase=pulse.pulse.relative_phase,
                )
                print("pulse", pulse.zhpulse)
                print("pulse", pulse.zhsweeper)
            elif any("duration" in param for param in parameters):
                exp.play(
                    signal=f"{section}{qubit.name}",
                    pulse=pulse.zhpulse,
                    length=pulse.zhsweeper,
                    phase=pulse.pulse.relative_phase,
                )
            # TODO: Join with "amplitude" or Zhsweeper_Line
            elif any("bias" in param for param in parameters):
                exp.play(
                    signal=f"{section}{qubit.name}",
                    pulse=pulse.zhpulse,
                    amplitude=pulse.zhsweeper,
                    phase=pulse.pulse.relative_phase,
                )
            elif partial_sweep.uid == "frequency" or partial_sweep.uid == "delay":
                exp.play(
                    signal=f"{section}{qubit.name}",
                    pulse=pulse.zhpulse,
                    phase=pulse.pulse.relative_phase,
                )
                print("pulse", pulse.zhpulse)

    # Create a similar section for Flux pulses if it works !!!
    # Flux on all qubits(Separete, Bias, Flux, Coupler)
    def Flux(self, exp, qubits):
        for qubit in qubits.values():
            if "c" in str(qubit.name):
                pass
            else:
                with exp.section(uid=f"sequence_bias{qubit.name}"):
                    i = 0
                    for pulse in self.sequence[f"flux{qubit.name}"]:
                        pulse.zhpulse.uid = pulse.zhpulse.uid + str(i)
                        if isinstance(pulse, ZhSweeper_line):
                            self.play_sweep(exp, qubit, pulse, section="flux")
                        else:
                            exp.play(signal=f"flux{qubit.name}", pulse=pulse.zhpulse)
                        i += 1

    # qubit drive pulses
    def Drive(self, exp, qubits):
        with exp.section(uid=f"sequence_drive"):
            for qubit in qubits.values():
                if "c" in str(qubit.name):
                    pass
                else:
                    time = 0
                    i = 0
                    if self.sequence[f"drive{qubit.name}"]:
                        for pulse in self.sequence[f"drive{qubit.name}"]:
                            # FIXME: Check delay schedule
                            if not isinstance(pulse, ZhSweeper_line):
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
                            elif isinstance(pulse, ZhSweeper_line):
                                exp.delay(signal=f"drive{qubit.name}", time=pulse.zhsweeper)

    # qubit readout pulse and data acquisition
    def Measure(self, exp, qubits):
        for qubit in qubits.values():
            if "c" in str(qubit.name):
                pass
            else:
                play_after = None
                # if f"drive{qubit.name}" in str(self.sequence):
                if self.sequence[f"drive{qubit.name}"]:
                    play_after = f"sequence_drive"
                with exp.section(uid=f"sequence_measure{qubit.name}", play_after=play_after):
                    if self.sequence[f"drive{qubit.name}"]:
                        last_drive_pulse = self.sequence[f"drive{qubit.name}"][-1]
                        time = round(last_drive_pulse.pulse.finish * 1e-9, 9)
                    else:
                        time = 0
                    i = 0
                    if self.sequence[f"readout{qubit.name}"]:
                        for pulse in self.sequence[f"readout{qubit.name}"]:
                            exp.delay(signal=f"measure{qubit.name}", time=0)
                            # exp.delay(signal=f"measure{qubit.name}", time= round(pulse.pulse.start * 1e-9, 9) - time)
                            time += round(pulse.pulse.duration * 1e-9, 9) + round(pulse.pulse.start * 1e-9, 9) - time
                            pulse.zhpulse.uid = pulse.zhpulse.uid + str(i)
                            if isinstance(pulse, ZhSweeper):
                                self.play_sweep(exp, qubit, pulse, section="measure")
                            else:
                                exp.play(
                                    signal=f"measure{qubit.name}", pulse=pulse.zhpulse, phase=pulse.pulse.relative_phase
                                )
                            # FIXME: This should be defined by the user as a pulse elsewhere
                            weight = lo.pulse_library.const(
                                uid="weight" + pulse.zhpulse.uid,
                                length=round(pulse.pulse.duration * 1e-9, 9),
                                amplitude=1.0,
                            )
                            exp.acquire(signal=f"acquire{qubit.name}", handle=f"sequence{qubit.name}", kernel=weight)
                            i += 1

    def qubit_reset(self, exp, qubits, relaxation_time, fast_reset=False):
        # relax time or fast reset after readout - for signal processing and qubit relaxation to ground state
        if fast_reset is not False:
            for qubit in qubits.values():
                if "c" in str(qubit.name):
                    pass
                else:
                    with exp.section(uid=f"fast_reset{qubit.name}", play_after=f"sequence_measure{qubit.name}"):
                        with exp.match_local(handle=f"acquire{qubit.name}"):
                            with exp.case(state=0):
                                pass
                            with exp.case(state=1):
                                exp.play(signal=f"drive{qubit.name}", pulse=ZhPulse(fast_reset[qubit.name]).zhpulse)
        else:
            for qubit in qubits.values():
                if "c" in str(qubit.name):
                    pass
                else:
                    with exp.section(uid=f"relax{qubit.name}", play_after=f"sequence_measure{qubit.name}"):
                        if qubit.flux is not None:
                            exp.reserve(signal=f"flux{qubit.name}")
                        if self.sequence[f"drive{qubit.name}"]:
                            exp.reserve(signal=f"drive{qubit.name}")
                        if self.sequence[f"readout{qubit.name}"]:
                            exp.delay(signal=f"measure{qubit.name}", time=relaxation_time)

    def sweep(self, qubits, sequence, *sweepers, nshots, relaxation_time, average=True, sim_time=10e-6):
        if relaxation_time is None:
            relaxation_time = self.relaxation_time

        self.sweepers = list(sweepers)
        self.sequence_zh(sequence, qubits, sweepers)
        self.calibration_step(qubits)
        self.create_exp(qubits, nshots, relaxation_time, fast_reset=False)
        self.run_exp()
        self.disconnect()

        # TODO: General, several readouts and qubits
        results = {}
        for qubit in qubits.values():
            if "c" in str(qubit.name):
                pass
            else:
                if self.sequence[f"readout{qubit.name}"]:
                    exp_res = self.results.get_data(f"sequence{qubit.name}")
                    i = exp_res.real
                    q = exp_res.imag
                    results[self.sequence[f"readout{qubit.name}"][0].pulse.serial] = ExecutionResults.from_components(
                        i, q
                    )

        self.run_sim(sim_time)
        # lo.show_pulse_sheet("pulses", self.exp) #FIXME: Wait until next LO release for bugs
        return results

    # TODO: Recursion tests and Better sweeps logic
    def sweep_recursion(self, qubits, exp, exp_calib, relaxation_time):
        sweeper = self.sweepers[-1]
        i = len(self.sweepers) - 1
        self.sweepers.remove(sweeper)

        parameter = None
        print("sweep", sweeper.parameter.name.lower())

        if sweeper.parameter.name.lower() == "frequency":
            for pulse in sweeper.pulses:
                qubit = pulse.qubit
                if pulse.type.name.lower() == "drive":
                    line = "drive"
                elif pulse.type.name.lower() == "readout":
                    line = "measure"
                exp_calib[f"{line}{qubit}"] = lo.SignalCalibration(
                    oscillator=lo.Oscillator(
                        frequency=ZhSweeper(pulse, sweeper, qubits[sweeper.pulses[0].qubit]).zhsweeper,
                        modulation_type=lo.ModulationType.HARDWARE,
                    )
                )

        if sweeper.parameter.name == "bias":
            for qubit in sweeper.qubits.values():
                # for qubit in sweeper.qubits.values():
                #     qubit = sweeper.qubits[qubit]

                parameter = ZhSweeper_line(sweeper, qubit, self.sequence_qibo).zhsweeper

        if sweeper.parameter.name == "delay":
            pulse = sweeper.pulses[0]
            qubit = pulse.qubit
            # for qubit in sweeper.qubits.values():
            #     qubit = sweeper.qubits[qubit]

            parameter = ZhSweeper_line(sweeper, qubit, self.sequence_qibo).zhsweeper

        if parameter is None:
            parameter = ZhSweeper(sweeper.pulses[0], sweeper, qubits[sweeper.pulses[0].qubit]).zhsweeper

        with exp.sweep(
            uid=f"sweep_{sweeper.parameter.name.lower()}_{i}",
            parameter=parameter,
            reset_oscillator_phase=False,
        ):
            if len(self.sweepers) > 0:
                self.sweep_recursion(qubits, exp, exp_calib, relaxation_time)
            else:
                self.select_exp(exp, qubits, relaxation_time, fast_reset=False)

    # -----------------------------------------------------------------------------

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


# Try to convert this to plotty
def plot_simulation(
    compiled_experiment,
    start_time=0.0,
    length=10e-6,
    xaxis_label="Time (s)",
    yaxis_label="Amplitude",
    plot_width=6,
    plot_height=2,
):
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
