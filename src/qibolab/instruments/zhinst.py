from collections import defaultdict

import laboneq.simple as lo
import numpy as np

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.result import ExecutionResults
from qibolab.sweeper import Parameter

# TODO: Add return clasified states.
# TODO: Simulation
# session = Session(device_setup=device_setup)
# session.connect(do_emulation=use_emulation)
# my_results = session.run(exp)


# TODO:For play Single shot,
# TODO:For sweep average true and false, single shot and cyclic.


# TODO: Add/Check for loops for multiple qubits
# TODO: Adapt(dont think I need it i I use lo.pulse_library and baking)
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
        return Zh_Sweeper


class Zurich(AbstractInstrument):
    def __init__(self, name, descriptor, use_emulation=False):
        self.name = name
        self.descriptor = descriptor
        self.emulation = use_emulation

        self.is_connected = False

        self.time_of_flight = 0
        self.smearing = 0

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
            self.relaxation_time = 10.0e-6
        else:
            self.relaxation_time = relaxation_time

        self.sequence_zh(sequence, sweepers=[])
        self.create_exp(qubits, nshots)
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
                    results[f"sequence{qubit.name}"] = ExecutionResults.from_components(i, q, shots)

        return results

    # def sequence_zh(self, sequence):
    #     Zhsequence = defaultdict(list)
    #     for pulse in sequence:
    #         Zhsequence[f"{pulse.type.name.lower()}{pulse.qubit}"].append(ZhPulse(pulse))
    #     self.sequence = Zhsequence

    def sequence_zh(self, sequence, sweepers):
        Zhsequence = defaultdict(list)
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
        self.sweeps = sweeps

    def create_exp(self, qubits, nshots, **Params):
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
                    # self.sweep_recursion(exp, self.sweeps[i], qubits, i)
                    sweeper = self.sweeps[i]
                    self.sweep_recursion()

                    with exp.sweep(uid=f"sweep_{sweeper.sweeper.parameter.name}_{i}", parameter=sweeper.zhsweeper):
                        # TODO: Check for line parameters and update calib
                        self.select_exp(exp, qubits)

                # if len(self.sweepers) == 1:
                #     if self.sweepers[0].parameter.name == "frequency":
                #         with exp.sweep(uid="sweep_param", parameter=self.sweepers_Zh[1]):

                #             self.select_exp(exp)

                #     qubit = self.sweepers[0].pulses[0].qubit
                #     # define experiment calibration - sweep over qubit drive frequency
                #     exp_calib = lo.Calibration()
                #     exp_calib[f"measure{qubit}"] = lo.SignalCalibration(
                #         oscillator=lo.Oscillator(
                #             frequency=self.sweepers_Zh[0],
                #             modulation_type=lo.ModulationType.HARDWARE,
                #         )
                #     )

                # exp.set_calibration(exp_calib)

            else:
                self.select_exp(exp, qubits)

            self.experiment = exp

            exp.set_signal_map(self.signal_map)

            self.experiment = exp

    def select_exp(self, exp, qubits):
        if "drive" in str(self.sequence):
            if "flux" in str(self.sequence):
                self.Flux(exp, qubits)
                self.Drive(exp, qubits)
            else:
                self.Drive(exp, qubits)
        elif "flux" in str(self.sequence):
            self.Flux(exp, qubits)
        self.Measure(exp, qubits)
        self.qubit_reset(exp, qubits)

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
                            self.play_sweep(exp, qubit, pulse)
                        else:
                            exp.play(signal=f"drive{qubit.name}", pulse=pulse.zhpulse, phase=pulse.pulse.relative_phase)
                            i += 1

    def play_sweep(self, exp, qubit, pulse):
        if pulse.sweeper.parameter.name == "amplitude":
            exp.play(
                signal=f"drive{qubit.name}",
                pulse=pulse.zhpulse,
                amplitude=pulse.zhsweeper,
                phase=pulse.pulse.relative_phase,
            )
        if pulse.sweeper.parameter.name == "length":
            exp.play(
                signal=f"drive{qubit.name}",
                pulse=pulse.zhpulse,
                length=pulse.zhsweeper,
                phase=pulse.pulse.relative_phase,
            )

    def Measure(self, exp, qubits):
        # qubit readout pulse and data acquisition
        with exp.section(uid=f"sequence_measure"):
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
                        exp.play(signal=f"measure{qubit.name}", pulse=pulse.zhpulse, phase=pulse.pulse.relative_phase)
                        weight = lo.pulse_library.const(
                            uid="weight" + pulse.zhpulse.uid,
                            length=round(pulse.pulse.duration * 1e-9, 9),
                            amplitude=1.0,
                        )
                        exp.acquire(signal=f"acquire{qubit.name}", handle=f"sequence{qubit.name}", kernel=weight)
                        i += 1

    def qubit_reset(self, exp, qubits, Fast_reset=False):
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
                        exp.delay(signal=f"measure{qubit.name}", time=self.relaxation_time)

    def sweep(self, qubits, sequence, *sweepers, nshots, relaxation_time, average=True):
        if relaxation_time is None:
            self.relaxation_time = 10.0e-6
        else:
            self.relaxation_time = relaxation_time

        self.sequence_zh(sequence, sweepers)
        self.create_exp(qubits, nshots)
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
                    results[f"sequence{qubit.name}"] = ExecutionResults.from_components(i, q, shots)

        return results

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


# ------------------------------------------------------------------------------------------------------------

# # Reload settings together if possible
# def reload_settings(self):
#     with open(self.runcard_file) as file:
#         self.settings = yaml.safe_load(file)
#     if self.is_connected:
#         self.setup(**self.settings)

# def apply_settings(self):
# self.def_calibration()
# self.Zsetup.set_calibration(self.calib)

# def sequences_to_ZurichPulses(self, sequences, sweepers=None):
#     self.sequences = sequences

#     sequence_Z_drives = []
#     sequence_Z_readouts = []
#     sequence_Z_weights = []
#     sequence_Z_fluxs = []
#     Delays = []
#     rel_phases = []
#     Drive_durations = []
#     addressed_qubits = []

#     Starts = []
#     Durations = []

#     for l in range(len(sequences)):
#         sequence = sequences[l]

#         sequence_Z_drive = []
#         sequence_Z_readout = []
#         sequence_Z_weight = []
#         sequence_Z_flux = []
#         starts = []
#         durations = []
#         rel_phase = []

#         i = 0
#         j = 0
#         k = 0
#         Drive_duration = 0

#         for pulse in sequence:
#             starts.append(pulse.start)
#             durations.append(pulse.duration)
#             rel_phase.append(pulse.relative_phase)

#             qubit = pulse.qubit
#             if qubit in addressed_qubits:
#                 pass
#             else:
#                 addressed_qubits.append(qubit)

#             if str(pulse.type) == "PulseType.DRIVE":
#                 if str(pulse.shape) == "Rectangular()":
#                     sequence_Z_drive.append(
#                         lo.pulse_library.const(
#                             uid=(f"drive_{qubit}_" + str(l) + "_" + str(i)),
#                             # length=pulse.duration * 1e-9,
#                             length=round(pulse.duration * 1e-9, 9),
#                             amplitude=pulse.amplitude,
#                         )
#                     )
#                 elif "Gaussian" in str(pulse.shape):
#                     sigma = str(pulse.shape).removeprefix("Gaussian(")
#                     sigma = float(sigma.removesuffix(")"))
#                     sequence_Z_drive.append(
#                         lo.pulse_library.gaussian(
#                             uid=(f"drive_{qubit}_" + str(l) + "_" + str(i)),
#                             # length=pulse.duration * 1e-9,
#                             length=round(pulse.duration * 1e-9, 9),
#                             amplitude=pulse.amplitude,
#                             sigma=2 / sigma,
#                         )
#                     )
#                 elif "Drag" in str(pulse.shape):
#                     params = str(pulse.shape).removeprefix("Drag(")
#                     params = params.removesuffix(")")
#                     params = params.split(",")
#                     sigma = float(params[0])
#                     beta = float(params[1])
#                     sequence_Z_drive.append(
#                         lo.pulse_library.drag(
#                             uid=(f"drive_{qubit}_" + str(l) + "_" + str(i)),
#                             # length=pulse.duration * 1e-9,
#                             length=round(pulse.duration * 1e-9, 9),
#                             amplitude=pulse.amplitude,
#                             sigma=2 / sigma,
#                             beta=beta,
#                             # beta=2 / beta,
#                         )
#                     )

#                 i += 1
#             if str(pulse.type) == "PulseType.READOUT":
#                 if str(pulse.shape) == "Rectangular()":
#                     sequence_Z_readout.append(
#                         lo.pulse_library.const(
#                             uid=(f"readout_{qubit}_" + str(l) + "_" + str(j)),
#                             # length=pulse.duration * 1e-9,
#                             length=round(pulse.duration * 1e-9, 9),
#                             amplitude=pulse.amplitude,
#                         )
#                     )

#                     sequence_Z_weight.append(
#                         lo.pulse_library.const(
#                             uid="readout_weighting_function" + str(l) + "_" + str(j),
#                             # length=pulse.duration * 1e-9,
#                             length=round(pulse.duration * 1e-9, 9),
#                             amplitude=1.0,
#                         )
#                     )
#                 j += 1

#             if str(pulse.type) == "PulseType.FLUX":
#                 if str(pulse.shape) == "Rectangular()":
#                     sequence_Z_flux.append(
#                         lo.pulse_library.const(
#                             uid=(f"flux_{qubit}_" + str(l) + "_" + str(k)),
#                             # length=pulse.duration * 1e-9,
#                             length=round(pulse.duration * 1e-9, 9),
#                             amplitude=pulse.amplitude,
#                         )
#                     )
#                 k += 1

#         delays = []
#         for i in range(len(starts) - 1):
#             delays.append(starts[i + 1] - durations[i])

#         Drive_durations.append(Drive_duration)
#         sequence_Z_fluxs.append(sequence_Z_flux)
#         sequence_Z_readouts.append(sequence_Z_readout)
#         sequence_Z_weights.append(sequence_Z_weight)
#         sequence_Z_drives.append(sequence_Z_drive)
#         Delays.append(delays)
#         rel_phases.append(rel_phase)

#         Starts.append(starts)
#         Durations.append(durations)

#     self.delays = Delays
#     self.sequence_drive = sequence_Z_drives
#     self.sequence_readout = sequence_Z_readouts
#     self.sequence_flux = sequence_Z_fluxs
#     self.sequence_weight = sequence_Z_weights
#     self.rel_phases = rel_phases
#     self.Drive_durations = Drive_durations
#     self.addressed_qubits = addressed_qubits

#     self.starts = Starts
#     self.durations = Durations

#     self.sweepers = sweepers
#     if sweepers != None:
#         self.sweepers = sweepers
#         sweepers_Zh = []
#         for sweep in sweepers:
#             sweepers_Zh.append(
#                 lo.LinearSweepParameter(
#                     uid=sweep.parameter.name,
#                     start=sweep.values[0],
#                     stop=sweep.values[-1],
#                     count=sweep.values.shape[0],
#                 )
#             )
#         self.sweepers_Zh = sweepers_Zh

# # TODO: 2 consecutive experiments without compiling (Needs memory fix)
# # TODO: Multiplex (For readout)
# # TODO: All the posible sweeps
# # TODO: Select averaging and acquisition modes
# def create_exp(self):
#     signals = []
#     if any(self.sequence_drive):
#         if any(self.sequence_flux):
#             for qubit in self.addressed_qubits:
#                 signals.append(lo.ExperimentSignal(f"drive{qubit}"))
#                 signals.append(lo.ExperimentSignal(f"flux{qubit}"))
#                 signals.append(lo.ExperimentSignal(f"measure{qubit}"))
#                 signals.append(lo.ExperimentSignal(f"acquire{qubit}"))
#         else:
#             for qubit in self.addressed_qubits:
#                 signals.append(lo.ExperimentSignal(f"drive{qubit}"))
#                 signals.append(lo.ExperimentSignal(f"measure{qubit}"))
#                 signals.append(lo.ExperimentSignal(f"acquire{qubit}"))
#     elif any(self.sequence_flux):
#         for qubit in self.addressed_qubits:
#             signals.append(lo.ExperimentSignal(f"flux{qubit}"))
#             signals.append(lo.ExperimentSignal(f"measure{qubit}"))
#             signals.append(lo.ExperimentSignal(f"acquire{qubit}"))
#     else:
#         for qubit in self.addressed_qubits:
#             signals.append(lo.ExperimentSignal(f"measure{qubit}"))
#             signals.append(lo.ExperimentSignal(f"acquire{qubit}"))

#     exp = lo.Experiment(
#         uid="Sequence",
#         signals=signals,
#     )

#     # for j in range(len(self.sequences)):
#     #     self.iteration = j

#     # For the Resonator Spec
#     with exp.acquire_loop_rt(
#         uid="shots",
#         count=self.settings["hardware_avg"],
#         # repetition_mode= lo.RepetitionMode.CONSTANT,
#         # repetition_time= 20e-6,
#         acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
#         # acquisition_type=lo.AcquisitionType.INTEGRATION,
#         averaging_mode=lo.AveragingMode.CYCLIC,
#         # averaging_mode=lo.AveragingMode.SINGLE_SHOT,
#     ):
#         # For multiplex readout
#         # with exp.acquire_loop_rt(
#         #     uid="shots",
#         #     count=self.settings["hardware_avg"],
#         #     # repetition_mode= lo.RepetitionMode.CONSTANT,
#         #     # repetition_time= 20e-6,
#         #     acquisition_type=lo.AcquisitionType.INTEGRATION,
#         #     averaging_mode=lo.AveragingMode.CYCLIC,
#         #     # averaging_mode=lo.AveragingMode.SINGLE_SHOT,
#         # ):

#         if self.sweepers is not None:
#             if len(self.sweepers) == 1:
#                 if self.sweepers[0].parameter.name == "frequency":
#                     with exp.sweep(parameter=self.sweepers_Zh[0]):
#                         k = 0

#                         self.select_exp(exp)

#                         qubit = self.sweepers[0].pulses[0].qubit
#                     # define experiment calibration - sweep over qubit drive frequency
#                     exp_calib = lo.Calibration()
#                     # exp_calib[f"{line}{qubit}"] = lo.SignalCalibration(
#                     exp_calib[f"measure{qubit}"] = lo.SignalCalibration(
#                         oscillator=lo.Oscillator(
#                             frequency=self.sweepers_Zh[0],
#                             modulation_type=lo.ModulationType.HARDWARE,
#                         )
#                     )

#                     exp.set_calibration(exp_calib)

#             elif len(self.sweepers) == 2:
#                 # for sweep in self.sweepers:

#                 # if sweep.parameter == "freq":

#                 self.amplitude = self.sweepers_Zh[1]
#                 with exp.sweep(
#                     uid="sweep_freq", parameter=self.sweepers_Zh[0], alignment=lo.SectionAlignment.RIGHT
#                 ):
#                     with exp.sweep(uid="sweep_param", parameter=self.sweepers_Zh[1]):
#                         k = 0

#                         self.select_exp(exp)

#                 qubit = self.sweepers[0].pulses[0].qubit
#                 # define experiment calibration - sweep over qubit drive frequency
#                 exp_calib = lo.Calibration()
#                 exp_calib[f"measure{qubit}"] = lo.SignalCalibration(
#                     oscillator=lo.Oscillator(
#                         frequency=self.sweepers_Zh[0],
#                         modulation_type=lo.ModulationType.HARDWARE,
#                     )
#                 )

#                 exp.set_calibration(exp_calib)

#             elif len(self.sweepers) == 5:
#                 # for sweep in self.sweepers:
#                 if self.sweepers[0].parameter == "freq_qs_0":
#                     with exp.sweep(
#                         uid="sweep_freq", parameter=self.sweepers_Zh, alignment=lo.SectionAlignment.RIGHT
#                     ):
#                         k = 0
#                         self.select_exp(exp)

#                     # qubit = self.sweepers[0].pulses[0].qubit
#                     exp_calib = lo.Calibration()
#                     for qubit in self.addressed_qubits:
#                         # define experiment calibration - sweep over qubit drive frequency
#                         exp_calib[f"drive{qubit}"] = lo.SignalCalibration(
#                             oscillator=lo.Oscillator(
#                                 frequency=self.sweepers_Zh[qubit],
#                                 modulation_type=lo.ModulationType.HARDWARE,
#                             )
#                         )

#                     exp.set_calibration(exp_calib)
#                 elif self.sweepers[0].parameter == "rabi_lenght_0" or self.sweepers[0].parameter == "rabi_amp_0":
#                     with exp.sweep(uid="sweep", parameter=self.sweepers_Zh, alignment=lo.SectionAlignment.RIGHT):
#                         self.select_exp(exp)

#         else:
#             self.select_exp(exp)

#     self.set_maps()
#     exp.set_signal_map(self.map_q)

#     self.experiment = exp

# def select_exp(self, exp):
#     if any(self.sequence_drive):
#         if any(self.sequence_flux):
#             for j in range(len(self.sequences)):
#                 self.iteration = j
#                 self.Flux(exp)
#                 self.Drive(exp)
#                 self.Measure(exp)
#                 self.qubit_reset(exp)

#         else:
#             for j in range(len(self.sequences)):
#                 self.iteration = j
#                 self.Drive(exp)
#                 # self.Drive_Rabi(exp)
#                 self.Measure(exp)
#                 self.qubit_reset(exp)

#     elif any(self.sequence_flux):
#         for j in range(len(self.sequences)):
#             self.iteration = j
#             self.Flux(exp)
#             self.Measure(exp)
#             self.qubit_reset(exp)

#     else:
#         for j in range(len(self.sequences)):
#             self.iteration = j
#             self.Measure(exp)
#             self.qubit_reset(exp)

# # Flux on all qubits
# def Flux(self, exp):
#     j = self.iteration
#     with exp.section(uid=f"sequence{j}_flux_bias", alignment=lo.SectionAlignment.RIGHT):
#         for pulse in self.sequence_flux[j]:
#             qubit = pulse.uid.split("_")[1]
#             exp.play(signal=f"flux{qubit}", pulse=pulse, amplitude=self.amplitude)
#         for qubit in self.addressed_qubits:
#             exp.delay(signal=f"flux{qubit}", time=self.settings["readout_delay"])

# def Drive(self, exp):
#     j = self.iteration
#     with exp.section(uid=f"sequence{j}_drive", alignment=lo.SectionAlignment.RIGHT):
#         i = 0
#         # for qubit in self.addressed_qubits:
#         # exp.delay(signal=f"drive{qubit}", time = 10e-9) #ramp up
#         for pulse in self.sequence_drive[j]:
#             qubit = pulse.uid.split("_")[1]
#             exp.play(signal=f"drive{qubit}", pulse=pulse, phase=self.rel_phases[j][i])
#             exp.reserve(signal=f"measure{qubit}")
#             if self.delays[j][i] > 0:
#                 qubit = pulse.uid.split("_")[1]
#                 exp.delay(signal=f"drive{qubit}", time=self.delays[j][i] * 1e-9)
#             i += 1

# def Drive_Rabi(self, exp):
#     j = self.iteration

#     with exp.section(uid=f"sequence{j}_drive", alignment=lo.SectionAlignment.RIGHT):
#         i = 0
#         # for qubit in self.addressed_qubits:
#         # exp.delay(signal=f"drive{qubit}", time = 10e-9) #ramp up
#         for pulse in self.sequence_drive[j]:
#             qubit = pulse.uid.split("_")[1]

#             if self.sweepers[0].parameter == "rabi_lenght_0":
#                 length = self.sweepers_Zh[int(qubit)]
#                 amp = pulse.amplitude
#             else:
#                 amp = self.sweepers_Zh[int(qubit)]
#                 length = pulse.length

#             exp.play(signal=f"drive{qubit}", pulse=pulse, phase=self.rel_phases[j][i], length=length, amplitude=amp)
#             exp.reserve(signal=f"measure{qubit}")
#             if self.delays[j][i] > 0:
#                 qubit = pulse.uid.split("_")[1]
#                 exp.delay(signal=f"drive{qubit}", time=self.delays[j][i] * 1e-9)
#             i += 1

# def Measure(self, exp):
#     # qubit readout pulse and data acquisition
#     j = self.iteration
#     with exp.section(uid=f"sequence{j}_measure"):
#         # with exp.section(uid=f"sequence{j}_measure", play_after=f"sequence{j}_drive"):
#         # exp.reserve("drive")
#         i = 0
#         for pulse in self.sequence_readout[j]:
#             qubit = pulse.uid.split("_")[1]
#             # FIXME: Lenght
#             exp.play(signal=f"measure{qubit}", pulse=pulse)
#             # exp.play(signal=f"measure{qubit}", pulse=pulse, phase=self.rel_phases[j][i], lenght = 1e-6)
#             # FIXME: Handle
#             if self.sweepers != None:
#                 if len(self.sweepers) == 2:
#                     exp.acquire(signal=f"acquire{qubit}", handle=f"sequence", kernel=self.sequence_weight[j][i])
#                 elif len(self.sweepers) == 5:
#                     exp.acquire(
#                         signal=f"acquire{qubit}", handle=f"sequence_{i}_{j}", kernel=self.sequence_weight[j][i]
#                     )
#                 elif len(self.sweepers) == 1:
#                     exp.acquire(signal=f"acquire{qubit}", handle=f"sequence_{j}", kernel=self.sequence_weight[j][i])

#             elif len(self.sequence_readout[j]) > 1:
#                 exp.acquire(signal=f"acquire{qubit}", handle=f"sequence_{i}_{j}", kernel=self.sequence_weight[j][i])
#             else:
#                 exp.acquire(signal=f"acquire{qubit}", handle=f"sequence_{j}", kernel=self.sequence_weight[j][i])
#             i += 1

# def qubit_reset(self, exp):
#     j = self.iteration
#     # relax time or fast reset after readout - for signal processing and qubit relaxation to ground state
#     if self.settings["Fast_reset"] == True:
#         for qubit in self.addressed_qubits:
#             with exp.section(uid=f"fast_reset", play_after=f"sequence{j}_measure"):
#                 with exp.match_local(handle=f"acquire{qubit}"):
#                     with exp.case(state=0):
#                         pass
#                         # exp.play(some_pulse)
#                     with exp.case(state=1):
#                         pass
#                         # exp.play(some_other_pulse)
#     else:
#         with exp.section(uid=f"relax_{j}", play_after=f"sequence{j}_measure"):
#             for qubit in self.addressed_qubits:
#                 exp.delay(signal=f"measure{qubit}", time=self.settings["readout_delay"])

# def sequence_to_Zurichpulses(self, sequence):
#     self.sequence = sequence
#     sequence_Z_drive = []
#     sequence_Z_readout = []
#     starts = []
#     durations = []
#     self.rel_phases = []
#     i = 0
#     j = 0
#     for pulse in sequence:
#         starts.append(pulse.start)
#         durations.append(pulse.duration)
#         # self.rel_phases.append(pulse.relative_phase)

#         if str(pulse.type) == "PulseType.DRIVE":
#             if str(pulse.shape) == "Rectangular()":
#                 sequence_Z_drive.append(
#                     lo.pulse_library.const(
#                         uid=("drive" + str(i)),
#                         length=pulse.duration * 1e-9,
#                         amplitude=pulse.amplitude,
#                     )
#                 )
#             elif "Gaussian" in str(pulse.shape):
#                 sigma = str(pulse.shape).removeprefix("Gaussian(")
#                 sigma = float(sigma.removesuffix(")"))
#                 sequence_Z_drive.append(
#                     lo.pulse_library.gaussian(
#                         uid=("drive" + str(i)),
#                         length=pulse.duration * 1e-9,
#                         amplitude=pulse.amplitude,
#                         sigma=2 / sigma,
#                     )
#                 )
#             elif "Drag" in str(pulse.shape):
#                 params = str(pulse.shape).removeprefix("Drag(")
#                 params = params.removesuffix(")")
#                 params = params.split(",")
#                 sigma = float(params[0])
#                 beta = float(params[1])
#                 sequence_Z_drive.append(
#                     lo.pulse_library.drag(
#                         uid=("drive" + str(i)),
#                         length=pulse.duration * 1e-9,
#                         amplitude=pulse.amplitude,
#                         sigma=2 / sigma,
#                         beta=beta,
#                         # beta=2 / beta,
#                     )
#                 )

#         i += 1
#         if str(pulse.type) == "PulseType.READOUT":
#             if str(pulse.shape) == "Rectangular()":
#                 sequence_Z_readout.append(
#                     lo.pulse_library.const(
#                         uid=("readout" + str(j)),
#                         length=pulse.duration * 1e-9,
#                         amplitude=pulse.amplitude,
#                     )
#                 )

#                 self.readout_weighting_function = lo.pulse_library.const(
#                     uid="readout_weighting_function",
#                     length=2 * pulse.duration * 1e-9,
#                     amplitude=1.0,
#                 )
#         j += 1

#     delays = []
#     for i in range(len(starts) - 1):
#         delays.append(starts[i + 1] - durations[i])

#     self.delays = delays
#     self.sequence_drive = sequence_Z_drive
#     self.sequence_readout = sequence_Z_readout

# # Separe play and sweep and add relax and shots.
# # And list of Qubits(Object)
# def execute_sequences(self, sequences, sweepers=None):
#     # if self.sequence == sequence:
#     #     self.repeat_seq()
#     # else:
#     #     self.sequence_to_ZurichPulses(sequence)
#     #     self.sequencePulses_to_exp()
#     #     self.run_seq()

#     self.sequences_to_ZurichPulses(sequences, sweepers)
#     self.create_exp()
#     self.run_seq()

#     spec_res = []
#     msr = []
#     phase = []
#     i = []
#     q = []

#     if self.sweepers != None:
#         if len(self.sweepers) == 2:
#             for j in range(self.sweepers[0].count):
#                 for k in range(self.sweepers[1].count):
#                     datapoint = self.results.get_data("sequence")[j][k]
#                     msr.append(abs(datapoint))
#                     phase.append(np.angle(datapoint))
#                     i.append(datapoint.real)
#                     q.append(datapoint.imag)

#             return msr, phase, i, q

#         elif len(self.sweepers) == 5:
#             for j in range(5):
#                 spec_res.append(self.results.get_data(f"sequence_{j}_{0}"))
#                 msr.append(abs(spec_res[j]))
#                 phase.append(np.angle(spec_res[j]))
#                 i.append(spec_res[j].real)
#                 q.append(spec_res[j].imag)

#             return msr, phase, i, q

#         elif len(self.sweepers) == 1:
#             # handles = result.result_handles
#             # results = {}
#             # for pulse in ro_pulses:
#             #     serial = pulse.serial
#             #     ires = handles.get(f"{serial}_I").fetch_all()
#             #     qres = handles.get(f"{serial}_Q").fetch_all()
#             #     if f"{serial}_shots" in handles:
#             #         shots = handles.get(f"{serial}_shots").fetch_all().astype(int)
#             #     else:
#             #         shots = None
#             #     results[pulse.qubit] = results[serial] = ExecutionResults.from_components(ires, qres, shots)
#             # return results

#             results = {}
#             for j in range(len(self.sequences)):
#                 for pulse in sequences[j].ro_pulses:
#                     spec_res = self.results.get_data(f"sequence_{j}")
#                     i = spec_res.real
#                     q = spec_res.imag

#                     # spec_res.append(self.results.get_data(f"sequence_{j}"))
#                     # msr.append(abs(spec_res[j]))
#                     # phase.append(np.angle(spec_res[j]))
#                     # i.append(spec_res[j].real)
#                     # q.append(spec_res[j].imag)

#             shots = self.settings["hardware_avg"]
#             results[pulse.qubit] = ExecutionResults.from_components(i, q, shots)

#             return results

#     elif len(self.sequence_readout[0]) > 1:
#         for j in range(len(self.sequences)):
#             for k in range(len(self.sequence_readout[j])):
#                 datapoint = self.results.get_data(f"sequence_{k}_{j}")
#                 msr.append(abs(datapoint))
#                 phase.append(np.angle(datapoint))
#                 i.append(datapoint.real)
#                 q.append(datapoint.imag)

#         return msr, phase, i, q

#     else:
#         for j in range(len(self.sequences)):
#             spec_res.append(self.results.get_data(f"sequence_{j}"))
#             msr.append(abs(spec_res[j]))
#             phase.append(np.angle(spec_res[j]))
#             i.append(spec_res[j].real)
#             q.append(spec_res[j].imag)

#         return msr, phase, i, q

# def sequencepulses_to_exp(self):
#     # Create Experiment

#     if len(self.sequence_drive) != 0:
#         exp = lo.Experiment(
#             uid="Sequence",
#             signals=[
#                 lo.ExperimentSignal("drive"),
#                 lo.ExperimentSignal("measure"),
#                 lo.ExperimentSignal("acquire"),
#             ],
#         )

#         ## experimental pulse sequence
#         # outer loop - real-time, cyclic averaging in standard integration mode
#         with exp.acquire_loop_rt(
#             uid="shots",
#             count=self.settings["hardware_avg"],
#             averaging_mode=lo.AveragingMode.SEQUENTIAL,
#             acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
#             # averaging_mode=lo.AveragingMode.CYCLIC,
#             # acquisition_type=lo.AcquisitionType.INTEGRATION,
#         ):
#             # # inner loop - real-time sweep of qubit drive pulse amplitude
#             # with exp.sweep(uid="sweep", parameter=sweep_rel_flat, alignment=SectionAlignment.RIGHT):
#             # qubit excitation - pulse amplitude will be swept
#             with exp.section(uid="qubit_excitation", alignment=lo.SectionAlignment.RIGHT):
#                 i = 0
#                 for pulse in self.sequence_drive:
#                     exp.play(signal="drive", pulse=pulse, phase=self.rel_phases[i])

#                     if self.delays[i] > 0:
#                         exp.delay(signal="drive", time=self.delays[i] * 1e-9)
#                     i += 1

#             # qubit readout pulse and data acquisition

#             with exp.section(uid="qubit_readout"):
#                 for pulse in self.sequence_readout:
#                     exp.reserve(signal="drive")

#                     exp.play(signal="measure", pulse=pulse, phase=self.rel_phases[i])

#                     integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]

#                     exp.acquire(signal="acquire", handle="sequence", length=integration_time)

#                     # exp.acquire(
#                     #     signal="acquire",
#                     #     handle="Sequence",
#                     #     kernel=self.readout_weighting_function,
#                     # )

#             # relax time after readout - for signal processing and qubit relaxation to ground state
#             with exp.section(uid="relax"):
#                 exp.delay(signal="measure", time=self.settings["readout_delay"])

#     # TODO: Add features of above to else
#     else:
#         exp = lo.Experiment(
#             uid="Sequence",
#             signals=[
#                 lo.ExperimentSignal("measure0"),
#                 lo.ExperimentSignal("acquire0"),
#             ],
#         )
#         ## experimental pulse sequence
#         # outer loop - real-time, cyclic averaging in standard integration mode
#         with exp.acquire_loop_rt(
#             uid="shots",
#             count=self.settings["hardware_avg"],
#             averaging_mode=lo.AveragingMode.CYCLIC,
#             acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
#             # acquisition_type=lo.AcquisitionType.INTEGRATION,
#         ):
#             # # inner loop - real-time sweep of qubit drive pulse amplitude
#             # qubit readout pulse and data acquisition

#             i = 0
#             with exp.section(uid="qubit_readout"):
#                 for pulse in self.sequence_readout:
#                     exp.play(signal="measure0", pulse=pulse)

#                     integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]

#                     exp.acquire(signal="acquire0", handle="sequence", length=integration_time)

#                     # exp.acquire(
#                     #     signal="acquire",
#                     #     handle="Sequence",
#                     #     kernel=self.readout_weighting_function,
#                     # )

#             # relax time after readout - for signal processing and qubit relaxation to ground state
#             with exp.section(uid="relax"):
#                 exp.delay(signal="measure0", time=self.settings["readout_delay"])

#     qubit = 0
#     map_q = {}
#     map_q[f"measure{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["measure_line"]
#     map_q[f"acquire{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["acquire_line"]

#     exp.set_signal_map(map_q)

#     self.experiment = exp

# def execute_pulse_sequence(self, sequence):
#     self.sequence_to_Zurichpulses(sequence)
#     self.sequencepulses_to_exp()
#     self.run_seq()

#     spec_res = self.results.get_data("sequence")

#     msr = abs(spec_res)
#     # phase = np.unwrap(np.angle(spec_res))
#     phase = np.angle(spec_res)
#     i = spec_res.real
#     q = spec_res.imag

#     return msr, phase, i, q

# def compile_exp(self, exp):
#     self.exp = self.session.compile(exp)

# def run_exp(self):
#     self.results = self.session.run(self.exp)

# def run_seq(self):
#     # compiler_settings = {
#     #     "SHFSG_FORCE_COMMAND_TABLE": True,
#     #     "SHFSG_MIN_PLAYWAVE_HINT": 32,
#     #     "SHFSG_MIN_PLAYZERO_HINT": 32,
#     # }

#     # self.exp = self.session.compile(self.experiment, compiler_settings=compiler_settings)

#     self.exp = self.session.compile(self.experiment)
#     self.results = self.session.run(self.exp, self.emulation)

# def run_multi(self):
#     compiler_settings = {
#         "SHFSG_FORCE_COMMAND_TABLE": True,
#         "SHFSG_MIN_PLAYWAVE_HINT": 32,
#         "SHFSG_MIN_PLAYZERO_HINT": 32,
#     }

#     self.exp = self.session.compile(self.experiment, compiler_settings=compiler_settings)
#     self.results = self.session.run(self.exp, self.emulation)

# def repeat_seq(self):
#     self.results = self.session.run(do_simulation=self.emulation)
