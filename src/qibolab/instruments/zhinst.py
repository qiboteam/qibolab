from collections import defaultdict

import laboneq.simple as lo
import numpy as np

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.result import ExecutionResults
from qibolab.sweeper import Parameter

# TODO: Simulation
# session = Session(device_setup=device_setup)
# session.connect(do_emulation=use_emulation)
# my_results = session.run(exp)

# TODO:For play Single shot
# TODO:For sweep average true and false, single shot and cyclic. (Select averaging and acquisition modes)

# FIXME: Multiplex (For readout)
# FIXME: Flux on all qubits
# FIXME: Lenght on pulses
# FIXME: Handle on acquires
# FIXME: I think is a hardware limitation but I cant sweep multiple drive oscillator at the same time
# FIXME: Docs & tests

# TODO:Add return clasified states.

###TEST
# TODO: Fast Reset
# TODO: Loops for multiple qubits [Parallel and Nested]

### NO NEED TO:
# TODO: Repeat last compiled experiment, is it useful ?
# def repeat_seq(self):
#     self.results = self.session.run()
# TODO: Do we need reload settings for some sweepers ? (I dont think so)
# def reload_settings(self):
#     with open(self.runcard_file) as file:
#         self.settings = yaml.safe_load(file)
#     if self.is_connected:
#         self.setup(**self.settings)
# def apply_settings(self):
# self.def_calibration()
# self.devicesetup.set_calibration(self.calib)

###EXTRA PARAMETERS I NEED:
# qubit.pi_pulse
# Soft_Mod : bool #For the readout lines
# fast_reset : bool


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


class ZhSweeper:
    def __init__(self, pulse, sweeper, qubit):
        self.sweeper = sweeper
        self.pulse = pulse

        # Need something better to store multiple sweeps on the same pulse
        self.zhsweeper = self.select_sweeper(sweeper, sweeper.parameter.name, qubit)
        self.zhsweepers = [self.select_sweeper(sweeper, sweeper.parameter.name, qubit)]

        self.signal = f"{pulse.type.name.lower()}{pulse.qubit}"
        self.zhpulse = ZhPulse(pulse).zhpulse

        # Stores the baking object (for pulses that need 1ns resolution)
        self.shot = None
        self.IQ = None
        self.shots = None
        self.threshold = None

    # Does LinearSweepParameter vs SweepParameter provide any advantage ?
    def select_sweeper(self, sweeper, parameter, qubit):
        if parameter == "amplitude":
            Zh_Sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values,
            )
        elif parameter == "duration":
            Zh_Sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values * 10e-9,
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
        elif parameter == "flux_bias":
            Zh_Sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values,
            )
        return Zh_Sweeper

    def add_sweeper(self, sweeper, qubit):
        self.zhsweepers.append(self.select_sweeper(sweeper, sweeper.parameter.name, qubit))


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

        # Improve the storing of multiple sweeps
        self.nsweeps = 0.0
        self.sweeps = None
        self.sweepers = None

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
                self.register_drive_line(
                    qubit=qubit, intermediate_frequency=qubit.drive_frequency - qubit.drive.local_oscillator.frequency
                )
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
            # FIXME: CANT MAKE IT WORK WITHOUT THIS !!!
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
                frequency=int(qubit.readout.local_oscillator.frequency),
            ),
            range=qubit.feedback.power_range,
            port_delay=None,
            delay_signal=0,
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

    def play(self, qubits, sequence, nshots, relaxation_time):
        if relaxation_time is None:
            relaxation_time = self.relaxation_time

        self.sequence_zh(sequence, qubits, sweepers=[])
        self.create_exp(qubits, nshots, relaxation_time)
        self.run_exp()

        # TODO: General, several readouts and qubits
        results = {}
        shots = 1024
        for qubit in qubits.values():
            if "c" in str(qubit.name):
                pass
            else:
                if self.sequence[f"readout{qubit.name}"]:
                    exp_res = self.results.get_data(f"sequence{qubit.name}")
                    i = exp_res.real
                    q = exp_res.imag
                    results[self.sequence[f"readout{qubit.name}"][0].pulse.serial] = ExecutionResults.from_components(
                        i, q, shots
                    )
        return results

    # TODO: Store the sweepers nice[Find a way to store nested vs parallel sweeps]
    def sequence_zh(self, sequence, qubits, sweepers):
        Zhsequence = defaultdict(list)
        for pulse in sequence:
            Zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"].append(ZhPulse(pulse))

        sweeps_all = {}
        sweeps = []
        nsweeps = 0
        for sweeper in sweepers:
            nsweeps += 1
            for pulse in sweeper.pulses:
                aux_list = Zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"]
                for element in aux_list:
                    if pulse == element.pulse:
                        if isinstance(aux_list[aux_list.index(element)], ZhPulse):
                            aux_list[aux_list.index(element)] = ZhSweeper(pulse, sweeper, qubits[pulse.qubit])
                        elif isinstance(aux_list[aux_list.index(element)], ZhSweeper):
                            aux_list[aux_list.index(element)].add_sweeper(sweeper, qubits[pulse.qubit])
                        sweeps.append(ZhSweeper(pulse, sweeper, qubits[pulse.qubit]))
        self.sequence = Zhsequence
        self.nsweeps = nsweeps  # Should count the dimension of the sweep 1D, 2D etc
        sweeps_all[nsweeps] = sweeps
        self.sweeps = sweeps_all  # Should count the number of sweeps taking into account parallel ones

    def create_exp(self, qubits, nshots, relaxation_time, **Params):
        signals = []
        for qubit in qubits.values():
            if "c" in str(qubit.name):
                signals.append(lo.ExperimentSignal(f"flux{qubit.name}"))
            else:
                signals.append(lo.ExperimentSignal(f"drive{qubit.name}"))
                if qubit.flux is not None:
                    signals.append(lo.ExperimentSignal(f"flux{qubit.name}"))
                signals.append(lo.ExperimentSignal(f"measure{qubit.name}"))
                signals.append(lo.ExperimentSignal(f"acquire{qubit.name}"))

        exp = lo.Experiment(
            uid="Sequence",
            signals=signals,
        )

        with exp.acquire_loop_rt(
            uid="shots",
            count=nshots,
            # repetition_mode= lo.RepetitionMode.CONSTANT,
            # repetition_time= 20e-6,
            # acquisition_type=lo.AcquisitionType.INTEGRATION,  # TODO: I dont know how to make it work
            acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
            # acquisition_type=lo.AcquisitionType.DISCRIMINATION,
            # averaging_mode=lo.AveragingMode.SEQUENTIAL,
            averaging_mode=lo.AveragingMode.CYCLIC,
            # averaging_mode=lo.AveragingMode.SINGLE_SHOT,
        ):
            if self.nsweeps > 0:
                exp_calib = lo.Calibration()
                self.sweep_recursion(qubits, exp, exp_calib, relaxation_time)
                exp.set_calibration(exp_calib)
            else:
                self.select_exp(exp, qubits, relaxation_time)
            self.experiment = exp
            exp.set_signal_map(self.signal_map)

    def select_exp(self, exp, qubits, relaxation_time):
        if "drive" in str(self.sequence):
            if "flux" in str(self.sequence):
                self.Flux(exp, qubits)
                self.Drive(exp, qubits)
            else:
                self.Drive(exp, qubits)
        elif "flux" in str(self.sequence):
            self.Flux(exp, qubits)
        self.Measure(exp, qubits)
        self.qubit_reset(exp, qubits, relaxation_time)

    def play_sweep(self, exp, qubit, pulse, section):
        # This loop for when a pulse is swept with several parameters(Max:3[Lenght, Amplitude, Phase])
        if len(self.sweeps) == 100:  # Need a better way of checking
            exp.play(
                signal=f"{section}{qubit.name}",
                pulse=pulse.zhpulse,
                amplitude=pulse.zhsweepers[0],
                length=pulse.zhsweepers[1],
                phase=pulse.pulse.relative_phase,
            )
        # This loop for when a pulse is swept with one or none parameters
        else:
            parameters = []
            for partial_sweep in pulse.zhsweepers:
                parameters.append(partial_sweep.uid)
            if any("amplitude" in param for param in parameters):
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
            else:
                # elif partial_sweep.uid == "frequency" or partial_sweep.uid == "flux_bias":
                exp.play(
                    signal=f"{section}{qubit.name}",
                    pulse=pulse.zhpulse,
                    phase=pulse.pulse.relative_phase,
                )

    # Flux on all qubits(Separete, Bias, Flux, Coupler)
    def Flux(self, exp, qubits):
        # with exp.section(uid=f"sequence_flux_bias", alignment=lo.SectionAlignment.RIGHT):
        with exp.section(uid=f"sequence_flux_bias"):
            for qubit in qubits.values():
                if "c" in str(qubit.name):
                    pass
                else:
                    time = 0
                    i = 0
                    for pulse in self.sequence[f"flux{qubit.name}"]:
                        exp.delay(signal=f"flux{qubit.name}", time=round(pulse.pulse.start * 1e-9, 9) - time)
                        time += round(pulse.pulse.duration * 1e-9, 9) + round(pulse.pulse.start * 1e-9, 9) - time
                        pulse.zhpulse.uid = pulse.zhpulse.uid + str(i)
                        if isinstance(pulse, ZhSweeper):
                            self.play_sweep(exp, qubit, pulse, section="flux")
                        else:
                            exp.play(signal=f"flux{qubit.name}", pulse=pulse.zhpulse)
                        i += 1

    def Drive(self, exp, qubits):
        with exp.section(uid=f"sequence_drive"):
            for qubit in qubits.values():
                if "c" in str(qubit.name):
                    pass
                else:
                    time = 0
                    i = 0
                    for pulse in self.sequence[f"drive{qubit.name}"]:
                        exp.delay(signal=f"drive{qubit.name}", time=round(pulse.pulse.start * 1e-9, 9) - time)
                        time += round(pulse.pulse.duration * 1e-9, 9) + round(pulse.pulse.start * 1e-9, 9) - time
                        pulse.zhpulse.uid = pulse.zhpulse.uid + str(i)
                        if isinstance(pulse, ZhSweeper):
                            self.play_sweep(exp, qubit, pulse, section="drive")
                        else:
                            exp.play(signal=f"drive{qubit.name}", pulse=pulse.zhpulse, phase=pulse.pulse.relative_phase)
                            i += 1

    # def Measure(self, exp, qubits):
    #     # qubit readout pulse and data acquisition
    #     play_after = None
    #     if "drive" in str(self.sequence):
    #         play_after = f"sequence_drive"
    #     with exp.section(uid=f"sequence_measure", play_after=play_after):
    #         for qubit in qubits.values():
    #             if "c" in str(qubit.name):
    #                 pass
    #             else:
    #                 if self.sequence[f"drive{qubit.name}"]:
    #                     last_drive_pulse = self.sequence[f"drive{qubit.name}"][-1]
    #                     time = round(last_drive_pulse.pulse.finish * 1e-9, 9)
    #                 else:
    #                     time = 0
    #                 i = 0
    #                 for pulse in self.sequence[f"readout{qubit.name}"]:
    #                     exp.delay(signal=f"measure{qubit.name}", time= round(pulse.pulse.start * 1e-9, 9) - time)
    #                     time += round(pulse.pulse.duration * 1e-9, 9) + round(pulse.pulse.start * 1e-9, 9) - time
    #                     pulse.zhpulse.uid = pulse.zhpulse.uid + str(i)
    #                     if isinstance(pulse, ZhSweeper):
    #                         self.play_sweep(exp, qubit, pulse, section="measure")
    #                     else:
    #                         exp.play(
    #                             signal=f"measure{qubit.name}", pulse=pulse.zhpulse, phase=pulse.pulse.relative_phase
    #                         )
    #                     # FIXME: This should be defined by the user as a pulse elsewhere
    #                     # Is this a qibo pulse ???
    #                     weight = lo.pulse_library.const(
    #                         uid="weight" + pulse.zhpulse.uid,
    #                         length=round(pulse.pulse.duration* 1e-9, 9),
    #                         amplitude=1.0,
    #                     )
    #                     exp.acquire(signal=f"acquire{qubit.name}", handle=f"sequence{qubit.name}", kernel=weight)
    #                     i += 1

    def Measure(self, exp, qubits):
        # qubit readout pulse and data acquisition
        for qubit in qubits.values():
            if "c" in str(qubit.name):
                pass
            else:
                play_after = None
                if "drive" in str(self.sequence):
                    play_after = f"sequence_drive"
                with exp.section(uid=f"sequence_measure{qubit.name}", play_after=play_after):
                    if self.sequence[f"drive{qubit.name}"]:
                        last_drive_pulse = self.sequence[f"drive{qubit.name}"][-1]
                        time = round(last_drive_pulse.pulse.finish * 1e-9, 9)
                    else:
                        time = 0
                    i = 0
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
                        # Is this a qibo pulse ???
                        weight = lo.pulse_library.const(
                            uid="weight" + pulse.zhpulse.uid,
                            length=round(pulse.pulse.duration * 1e-9, 9),
                            amplitude=1.0,
                        )
                        exp.acquire(signal=f"acquire{qubit.name}", handle=f"sequence{qubit.name}", kernel=weight)
                        i += 1

    def qubit_reset(self, exp, qubits, relaxation_time, Fast_reset=False):
        # relax time or fast reset after readout - for signal processing and qubit relaxation to ground state
        if Fast_reset:
            for qubit in qubits.values():
                if "c" in str(qubit.name):
                    pass
                else:
                    with exp.section(uid=f"fast_reset", play_after=f"sequence_measure"):
                        with exp.match_local(handle=f"acquire{qubit.name}"):
                            with exp.case(state=0):
                                pass
                            with exp.case(state=1):
                                exp.play(ZhPulse(qubit.pi_pulse).zhpulse)
        else:
            # with exp.section(uid=f"relax", play_after=f"sequence_measure"):
            #     for qubit in qubits.values():
            #         if "c" in str(qubit.name):
            #             pass
            #         else:
            #             if qubit.flux is not None:
            #                 exp.reserve(signal=f"flux{qubit.name}")
            #             exp.reserve(signal=f"drive{qubit.name}")
            #             exp.delay(signal=f"measure{qubit.name}", time=relaxation_time)
            for qubit in qubits.values():
                if "c" in str(qubit.name):
                    pass
                else:
                    with exp.section(uid=f"relax{qubit.name}", play_after=f"sequence_measure{qubit.name}"):
                        if qubit.flux is not None:
                            exp.reserve(signal=f"flux{qubit.name}")
                        exp.reserve(signal=f"drive{qubit.name}")
                        exp.delay(signal=f"measure{qubit.name}", time=relaxation_time)

    def sweep(self, qubits, sequence, *sweepers, nshots, relaxation_time, average=True):
        if relaxation_time is None:
            relaxation_time = self.relaxation_time

        self.sweepers = list(sweepers)
        self.sequence_zh(sequence, qubits, sweepers)
        self.create_exp(qubits, nshots, relaxation_time)
        self.run_exp()

        # TODO: General, several readouts and qubits
        results = {}
        shots = nshots
        for qubit in qubits.values():
            if "c" in str(qubit.name):
                pass
            else:
                if self.sequence[f"readout{qubit.name}"]:
                    exp_res = self.results.get_data(f"sequence{qubit.name}")
                    i = exp_res.real
                    q = exp_res.imag
                    results[self.sequence[f"readout{qubit.name}"][0].pulse.serial] = ExecutionResults.from_components(
                        i, q, shots
                    )

        lo.show_pulse_sheet("pulses", self.exp)
        return results

    # TODO: Recursion tests and Better sweeps logic
    def sweep_recursion(self, qubits, exp, exp_calib, relaxation_time):
        sweeper = self.sweepers[-1]
        i = len(self.sweepers) - 1
        self.sweepers.remove(sweeper)

        print(sweeper.parameter.name.lower())
        print(ZhSweeper(sweeper.pulses[0], sweeper, qubits[sweeper.pulses[0].qubit]).zhsweeper)

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

        with exp.sweep(
            uid=f"sweep_{sweeper.parameter.name.lower()}_{i}",
            parameter=ZhSweeper(sweeper.pulses[0], sweeper, qubits[sweeper.pulses[0].qubit]).zhsweeper,
            reset_oscillator_phase=False,
        ):
            if len(self.sweepers) > 0:
                self.sweep_recursion(qubits, exp, exp_calib, relaxation_time)
            else:
                self.select_exp(exp, qubits, relaxation_time)
