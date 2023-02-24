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

# TODO: Add return clasified states.
# TODO:For play Single shot,
# TODO:For sweep average true and false, single shot and cyclic. (Select averaging and acquisition modes)

# TODO: Add/Check for loops for multiple qubits
# TODO: Baking(se lo.pulse_library and baking) using sampled pulsed at 2GSa/s
# sampled pulses are not parameterized. However, their length and phase can be swept

# TODO: Do we need reload settings for some sweepers ? (I dont think so)
# # Reload settings together if possible
# def reload_settings(self):
#     with open(self.runcard_file) as file:
#         self.settings = yaml.safe_load(file)
#     if self.is_connected:
#         self.setup(**self.settings)
# def apply_settings(self):
# self.def_calibration()
# self.devicesetup.set_calibration(self.calib)

# TODO: Repeat last compiled experiment, is it useful ?
# def repeat_seq(self):
#     self.results = self.session.run()
# # TODO: All the posible sweeps
# # TODO: Add 2d sweeps

# # FIXME: Multiplex (For readout)
# FIXME: Flux on all qubits
## FIXME: Lenght on pulses
# FIXME: Handle on acquires
# FIXME: I think is a hardware limitation but I cant sweep multple drive oscillator at the same time
# FIXME: Docs & tests


class ZhPulse:
    def __init__(self, pulse):
        self.pulse = pulse
        self.signal = f"{pulse.type.name.lower()}{pulse.qubit}"
        self.zhpulse = self.select_pulse(pulse, pulse.type.name.lower())

        # Stores the baking object (for pulses that need less than .5ns resolution)
        self.baked = None
        self.baked_amplitude = None

        self.shot = None
        self.IQ = None
        self.shots = None
        self.threshold = None

    def bake(self, config):
        pass

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
    def __init__(self, pulse, sweeper):
        self.sweeper = sweeper
        self.zhsweeper = self.select_sweeper(sweeper, sweeper.parameter.name)

        self.pulse = pulse
        self.signal = f"{pulse.type.name.lower()}{pulse.qubit}"
        self.zhpulse = ZhPulse(pulse).zhpulse

        # Stores the baking object (for pulses that need 1ns resolution)
        self.shot = None
        self.IQ = None
        self.shots = None
        self.threshold = None

    def select_sweeper(self, sweeper, parameter):
        if parameter == "amplitude":
            Zh_Sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values,
            )
        elif parameter == "length":
            Zh_Sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values * 10e-9,
            )
        elif parameter == "frequency":
            sweeper.values

            Zh_Sweeper = lo.LinearSweepParameter(
                uid=sweeper.parameter.name,
                start=sweeper.values[0],
                stop=sweeper.values[-1],
                count=len(sweeper.values),
            )
        elif parameter == "flux_bias":
            Zh_Sweeper = lo.SweepParameter(
                uid=sweeper.parameter.name,
                values=sweeper.values,
            )
        return Zh_Sweeper


class Zurich(AbstractInstrument):
    def __init__(self, name, descriptor, use_emulation=False):
        self.name = name
        self.descriptor = descriptor
        self.emulation = use_emulation

        self.is_connected = False

        self.time_of_flight = 0
        self.smearing = 0

        self.signal_map = {}
        self.calibration = lo.Calibration()

        self.device_setup = None
        self.session = None
        self.device = None

        self.relaxation_time = 0.0
        self.time_of_flight = 0.0
        self.smearing = 0.0

        self.exp = None
        self.experiment = None
        self.results = None

        self.sequence = None
        self.nsweps = 0.0
        self.sweeps = None

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
                    # self.device.reset()
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
        self.smearing = smearing

        self.signal_map = {}
        self.calibration = lo.Calibration()

        # Map can be done in experiment, but general calibration needs to be here
        # General calibration != experiment calibration
        for qubit in qubits.values():
            if "c" in str(qubit.name):
                self.register_flux_line(qubit)
            else:
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

        self.sequence_zh(sequence, sweepers=[])
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

    # TODO: Store the sweepers nice
    def sequence_zh(self, sequence, sweepers):
        Zhsequence = defaultdict(list)
        sweeps_all = {}
        sweeps = []
        for pulse in sequence:
            Zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"].append(ZhPulse(pulse))
        nsweps = 0
        for sweeper in sweepers:
            nsweps += 1
            for pulse in sweeper.pulses:
                aux_list = Zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"]
                for element in aux_list:
                    if pulse == element.pulse:
                        aux_list[aux_list.index(element)] = ZhSweeper(pulse, sweeper)
                        sweeps.append(ZhSweeper(pulse, sweeper))
        self.sequence = Zhsequence
        self.nsweps = nsweps
        sweeps_all[nsweps] = sweeps
        self.sweeps = sweeps_all

    def create_exp(self, qubits, nshots, relaxation_time, **Params):
        signals = []

        for qubit in qubits.values():
            if "c" in str(qubit.name):
                signals.append(lo.ExperimentSignal(f"flux{qubit.name}"))
            else:
                signals.append(lo.ExperimentSignal(f"drive{qubit.name}"))
                signals.append(lo.ExperimentSignal(f"flux{qubit.name}"))
                signals.append(lo.ExperimentSignal(f"measure{qubit.name}"))
                signals.append(lo.ExperimentSignal(f"acquire{qubit.name}"))

        exp = lo.Experiment(
            uid="Sequence",
            signals=signals,
        )

        # For the Resonator Spec
        # with exp.acquire_loop_rt(
        #     uid="shots",
        #     count=nshots,
        #     # repetition_mode= lo.RepetitionMode.CONSTANT,
        #     # repetition_time= 20e-6,
        #     # acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
        #     acquisition_type=lo.AcquisitionType.INTEGRATION,
        #     averaging_mode=lo.AveragingMode.CYCLIC,
        #     # averaging_mode=lo.AveragingMode.SINGLE_SHOT,
        # ):
        # For multiplex readout
        with exp.acquire_loop_rt(
            uid="shots",
            count=nshots,
            # repetition_mode= lo.RepetitionMode.CONSTANT,
            # repetition_time= 20e-6,
            acquisition_type=lo.AcquisitionType.INTEGRATION,
            # acquisition_type=lo.AcquisitionType.DISCRIMINATION,
            averaging_mode=lo.AveragingMode.CYCLIC,
            # averaging_mode=lo.AveragingMode.SINGLE_SHOT,
        ):
            # if isinstance(self.sequence["drive4"][0], ZhSweeper):

            if self.nsweps > 0:
                for i in range(len(self.sweeps)):
                    # # TODO: Recursion
                    # self.sweep_recursion(exp, self.sweeps[i], qubits, i)

                    sweeper = self.sweeps[i + 1]
                    parameters = []
                    for sweep in sweeper:
                        parameters.append(sweep.zhsweeper)

                    with exp.sweep(
                        uid=f"sweep_{sweeper[0].sweeper.parameter.name}_{i}",
                        parameter=parameters,
                        reset_oscillator_phase=False,
                    ):
                        self.select_exp(exp, qubits, relaxation_time)
                    if sweeper[0].sweeper.parameter.name == "frequency":
                        exp_calib = lo.Calibration()
                        for sweep in sweeper:
                            qubit = sweep.pulse.qubit
                            if sweep.pulse.type.name.lower() == "drive":
                                line = "drive"
                            elif sweep.pulse.type.name.lower() == "readout":
                                line = "measure"
                            exp_calib[f"{line}{qubit}"] = lo.SignalCalibration(
                                oscillator=lo.Oscillator(
                                    frequency=sweep.zhsweeper,
                                    modulation_type=lo.ModulationType.HARDWARE,
                                )
                            )
                    elif sweeper[0].sweeper.parameter.name == "flux_bias":
                        exp_calib = lo.Calibration()
                        for sweep in sweeper:
                            qubit = sweep.pulse.qubit
                            exp_calib[f"flux{qubit}"] = lo.SignalCalibration(voltage_offset=sweep.zhsweeper)
                exp.set_calibration(exp_calib)
            else:
                self.select_exp(exp, qubits, relaxation_time)
            self.experiment = exp
            exp.set_signal_map(self.signal_map)
            self.experiment = exp

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
        if pulse.sweeper.parameter.name == "amplitude":
            exp.play(
                signal=f"{section}{qubit.name}",
                pulse=pulse.zhpulse,
                amplitude=pulse.zhsweeper,
                phase=pulse.pulse.relative_phase,
            )
        if pulse.sweeper.parameter.name == "length":
            exp.play(
                signal=f"{section}{qubit.name}",
                pulse=pulse.zhpulse,
                length=pulse.zhsweeper,
                phase=pulse.pulse.relative_phase,
            )
        if pulse.sweeper.parameter.name == "frequency" or pulse.sweeper.parameter.name == "flux_bias":
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

    # TODO: Analyze the two options for the sections
    # their parallel sweeper cabilities or multiplexed readout
    # def Drive(self, exp, qubits):
    #     for qubit in qubits.values():
    #         if "c" in str(qubit.name):
    #             pass
    #         else:
    #             with exp.section(uid=f"sequence_drive{qubit.name}"):
    #                 time = 0
    #                 i = 0
    #                 for pulse in self.sequence[f"drive{qubit.name}"]:
    #                     exp.delay(signal=f"drive{qubit.name}", time=round(pulse.pulse.start * 1e-9, 9) - time)
    #                     time += round(pulse.pulse.duration * 1e-9, 9) + round(pulse.pulse.start * 1e-9, 9) - time
    #                     pulse.zhpulse.uid = pulse.zhpulse.uid + str(i)
    #                     if isinstance(pulse, ZhSweeper):
    #                         self.play_sweep(exp, qubit, pulse, section="drive")
    #                     else:
    #                         exp.play(signal=f"drive{qubit.name}", pulse=pulse.zhpulse, phase=pulse.pulse.relative_phase)
    #                         i += 1

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
    #     for qubit in qubits.values():
    #         if "c" in str(qubit.name):
    #             pass
    #         else:
    #             with exp.section(uid=f"sequence_measure{qubit.name}", play_after=f"sequence_drive{qubit.name}"):
    #                 time = 0
    #                 i = 0
    #                 for pulse in self.sequence[f"readout{qubit.name}"]:
    #                     exp.delay(signal=f"measure{qubit.name}", time=round(pulse.pulse.start * 1e-9, 9) - time)
    #                     exp.delay(signal=f"acquire{qubit.name}", time=round(pulse.pulse.start * 1e-9, 9) - time)
    #                     time += round(pulse.pulse.duration * 1e-9, 9) + round(pulse.pulse.start * 1e-9, 9) - time
    #                     pulse.zhpulse.uid = pulse.zhpulse.uid + str(i)
    #                     if isinstance(pulse, ZhSweeper):
    #                         self.play_sweep(exp, qubit, pulse, section="measure")
    #                     else:
    #                         exp.play(
    #                             signal=f"measure{qubit.name}", pulse=pulse.zhpulse, phase=pulse.pulse.relative_phase
    #                         )
    #                     weight = lo.pulse_library.const(
    #                         uid="weight" + pulse.zhpulse.uid,
    #                         length=round(pulse.pulse.duration * 1e-9, 9),
    #                         amplitude=1.0,
    #                     )
    #                     exp.acquire(signal=f"acquire{qubit.name}", handle=f"sequence{qubit.name}", kernel=weight)
    #                     i += 1

    def Measure(self, exp, qubits):
        # qubit readout pulse and data acquisition
        with exp.section(uid=f"sequence_measure", play_after=f"sequence_drive"):
            for qubit in qubits.values():
                if "c" in str(qubit.name):
                    pass
                else:
                    time = 0
                    i = 0
                    for pulse in self.sequence[f"readout{qubit.name}"]:
                        exp.delay(signal=f"measure{qubit.name}", time=round(pulse.pulse.start * 1e-9, 9) - time)
                        exp.delay(signal=f"acquire{qubit.name}", time=round(pulse.pulse.start * 1e-9, 9) - time)
                        time += round(pulse.pulse.duration * 1e-9, 9) + round(pulse.pulse.start * 1e-9, 9) - time
                        pulse.zhpulse.uid = pulse.zhpulse.uid + str(i)
                        if isinstance(pulse, ZhSweeper):
                            self.play_sweep(exp, qubit, pulse, section="measure")
                        else:
                            exp.play(
                                signal=f"measure{qubit.name}", pulse=pulse.zhpulse, phase=pulse.pulse.relative_phase
                            )
                        weight = lo.pulse_library.const(
                            uid="weight" + pulse.zhpulse.uid,
                            length=round(pulse.pulse.duration * 1e-9, 9),
                            amplitude=1.0,
                        )
                        exp.acquire(signal=f"acquire{qubit.name}", handle=f"sequence{qubit.name}", kernel=weight)
                        i += 1

    # TODO: Hardware clasification(lo.acquisitionMode.Driscrimination) and fast reset
    # def qubit_reset(self, exp, qubits, relaxation_time, Fast_reset=False):
    #     # relax time or fast reset after readout - for signal processing and qubit relaxation to ground state
    #     if Fast_reset:
    #         for qubit in qubits.values():
    #             if "c" in str(qubit.name):
    #                 pass
    #             else:
    #                 with exp.section(uid=f"fast_reset", play_after=f"sequence_measure"):
    #                     with exp.match_local(handle=f"acquire{qubit}"):
    #                         with exp.case(state=0):
    #                             pass
    #                             # exp.play(some_pulse)
    #                         with exp.case(state=1):
    #                             pass
    #                             # exp.play(some_other_pulse)
    #     else:
    #         for qubit in qubits.values():
    #             if "c" in str(qubit.name):
    #                 pass
    #             else:
    #                 with exp.section(uid=f"relax{qubit.name}", play_after=f"sequence_measure{qubit.name}"):
    #                     exp.reserve(signal=f"flux{qubit.name}")
    #                     exp.reserve(signal=f"drive{qubit.name}")
    #                     exp.delay(signal=f"measure{qubit.name}", time=relaxation_time)

    def qubit_reset(self, exp, qubits, relaxation_time, Fast_reset=False):
        # relax time or fast reset after readout - for signal processing and qubit relaxation to ground state
        if Fast_reset:
            for qubit in qubits.values():
                if "c" in str(qubit.name):
                    pass
                else:
                    with exp.section(uid=f"fast_reset", play_after=f"sequence_measure"):
                        with exp.match_local(handle=f"acquire{qubit}"):
                            with exp.case(state=0):
                                pass
                                # exp.play(some_pulse)
                            with exp.case(state=1):
                                pass
                                # exp.play(some_other_pulse)
        else:
            with exp.section(uid=f"relax", play_after=f"sequence_measure"):
                for qubit in qubits.values():
                    if "c" in str(qubit.name):
                        pass
                    else:
                        exp.reserve(signal=f"flux{qubit.name}")
                        exp.reserve(signal=f"drive{qubit.name}")
                        exp.delay(signal=f"measure{qubit.name}", time=relaxation_time)

    def sweep(self, qubits, sequence, *sweepers, nshots, relaxation_time, average=True):
        if relaxation_time is None:
            relaxation_time = self.relaxation_time

        self.sequence_zh(sequence, sweepers)
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

    # TODO: recursion
    def sweep_recursion(self, exp, sweepers, qubits, i):
        pass
        # with exp.sweep(uid=f"sweep_{sweeper.parameter.name}_{i}", parameter=sweeper.zhsweeper):
        # remove used sweeper
        #     if sweeper remaining:
        #         self.sweep_recursion()
        # --------------

        # if sweeper.pulses is not None:
        #     if sweeper.parameter is Parameter.frequency:

        #         freqs0 = []
        #         for pulse in sweeper.pulses:
        #             qubit = qubits[pulse.qubit]
        #             if pulse.type is PulseType.DRIVE:
        #                 lo_frequency = int(qubit.drive.local_oscillator.frequency)
        #             elif pulse.type is PulseType.READOUT:
        #                 lo_frequency = int(qubit.readout.local_oscillator.frequency)
        #             else:
        #                 raise_error(NotImplementedError, f"Cannot sweep frequency of pulse of type {pulse.type}.")
        #             # convert to IF frequency for readout and drive pulses

        #             if len(sweepers) > 1:
        #                 self.sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)

        #     elif sweeper.parameter is Parameter.amplitude:

        #         with for_(*from_array(a, sweeper.values)):
        #             for pulse in sweeper.pulses:

        #                 if len(sweepers) > 1:
        #                     self.sweep_recursion(sweepers[1:], qubits, qmsequence, relaxation_time)
        #                 else:
        #                     self.play_pulses(qmsequence)

        #     elif sweeper.qubits is not None:
        #         if sweeper.parameter is Parameter.bias:
        #             pass
        #         else:
        #             raise_error(NotImplementedError, "Sweeper configuration not implemented.")

        # else:
        #     raise_error(NotImplementedError, "Sweeper configuration not implemented.")
