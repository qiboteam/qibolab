import laboneq.simple as lo
import matplotlib.pyplot as plt
import numpy as np
import yaml

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.result import ExecutionResults

# TODO: Pulses timing
# TODO: se.finish and play with this for pulses
# TODO: Add return clasified states.

# TODO: Sweeps on Amplitude and not on range, but change it on top on them

# TODO: Simulation
# session = Session(device_setup=device_setup)
# session.connect(do_emulation=use_emulation)

# my_results = session.run(exp, do_simulation=True)

# TODO: Add/Check for loops for multiple qubits


# TODO: Adapt( dont think I need it i I use lo.pulse_library)
class ZhPulse:
    def __init__(self, pulse):
        self.pulse = pulse
        self.signal = f"{pulse.type.name.lower()}{pulse.qubit}"
        self.operation = pulse.serial
        self.zhpulse = self.select_pulse(pulse, pulse.type.name.lower())
        self.relative_phase = pulse.relative_phase / (2 * np.pi)

        # Stores the baking object (for pulses that need 1ns resolution)
        self.baked = None
        self.baked_amplitude = None

        self.shot = None
        self.I_st = None
        self.Q_st = None
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


# TODO: Adapt
class ZhSequence(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # keep track of readout pulses for registering outputs
        self.ro_pulses = []
        # map from qibolab pulses to ZhPulses (useful when sweeping)
        self.pulse_to_zhpulse = {}

    def add(self, pulse):
        # if not isinstance(pulse, Pulse):
        #     raise_error(TypeError, f"Pulse {pulse} has invalid type {type(pulse)}.")

        zhpulse = ZhPulse(pulse)

        self.pulse_to_zhpulse[pulse.serial] = zhpulse
        if pulse.type.name == "READOUT":
            self.ro_pulses.append(zhpulse)
        super().append(zhpulse)
        return zhpulse


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
                    # self.session = lo.Session(self.Zsetup)
                    # self.device = self.session.connect(self.address)
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

    # FIXME: What are these for ???
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

        for qubit in qubits.values():
            if qubit.flux:
                self.register_flux_line(qubit)

    def register_flux_line(self, qubit):
        """Registers qubit flux line to calibration and signal map."""
        q = qubit.name
        self.signal_map[f"flux{q}"] = self.device_setup.logical_signal_groups[f"q{q}"].logical_signals["flux_line"]
        self.calibration[f"/logical_signal_groups/q{q}/flux_line"] = lo.SignalCalibration(
            range=qubit.flux.power_range, port_delay=0, delay_signal=0, voltage_offset=qubit.flux.offset
        )

    def compile_exp(self, exp):
        self.exp = self.session.compile(exp)

    def run_exp(self):
        self.results = self.session.run(self.exp)

    def run_seq(self):
        # Compiler settings required for active reset and multiplex.
        compiler_settings = {
            "SHFSG_FORCE_COMMAND_TABLE": True,
            "SHFSG_MIN_PLAYWAVE_HINT": 32,
            "SHFSG_MIN_PLAYZERO_HINT": 32,
        }

        self.exp = self.session.compile(self.experiment, compiler_settings=compiler_settings)

        self.exp = self.session.compile(self.experiment)
        self.results = self.session.run(self.exp, self.emulation)

    def play(self, qubits, sequence, nshots, relaxation_time):
        if relaxation_time is None:
            self.relaxation_time = 10.0e-6
        else:
            self.relaxation_time = relaxation_time

        self.sequences_to_ZurichPulses(sequence)
        self.create_exp()
        self.run_seq()

        results = {}
        for j in range(len(self.sequences)):
            spec_res = self.results.get_data(f"sequence_{j}")
            i = spec_res.real
            q = spec_res.imag

        shots = 1024
        results[self.sequence_readout[0].qubit] = ExecutionResults.from_components(i, q, shots)

        return results

    def select_pulse(self, pulse, type, order):
        if str(pulse.shape) == "Rectangular()":
            Zh_Pulse = lo.pulse_library.const(
                uid=(f"{type}_{pulse.qubit}_" + str(order[0]) + "_" + str(order[1])),
                length=round(pulse.duration * 1e-9, 9),
                amplitude=pulse.amplitude,
            )
        elif "Gaussian" in str(pulse.shape):
            sigma = pulse.shape.rel_sigma
            Zh_Pulse = lo.pulse_library.gaussian(
                uid=(f"{type}_{pulse.qubit}_" + str(order[0]) + "_" + str(order[1])),
                length=round(pulse.duration * 1e-9, 9),
                amplitude=pulse.amplitude,
                sigma=2 / sigma,
            )
        elif "Drag" in str(pulse.shape):
            sigma = pulse.shape.rel_sigma
            beta = pulse.shape.beta
            Zh_Pulse = lo.pulse_library.drag(
                uid=(f"{type}_{pulse.qubit}_" + str(order[0]) + "_" + str(order[1])),
                length=round(pulse.duration * 1e-9, 9),
                amplitude=pulse.amplitude,
                sigma=2 / sigma,
                beta=beta,
            )
        return Zh_Pulse

    def sequence_zh(self, sequence):
        Zhsequence = {}
        l, i, j, k, m, r = 0, 0, 0, 0, 0, 0
        for pulse in sequence:
            Zhsequence[pulse.serial] = ZhPulse(pulse)
        self.sequence = Zhsequence

    # def register_pulse(self, pulse):
    #     Zhsequence = {}
    #     l, i, j, k, m, r = 0,0,0,0,0,0
    #     if pulse.type is PulseType.DRIVE:
    #         Zhsequence.append(self.select_pulse(pulse, "Drive", order = [l,i]))
    #         ZhPulse(pulse)
    #         i += 1
    #     elif pulse.type is PulseType.READOUT:
    #         Zhsequence.append(self.select_pulse(pulse, "Readout", order = [l,j]))
    #         j += 1
    #     elif pulse.type is PulseType.FLUX:
    #         Zhsequence.append(self.select_pulse(pulse, "Flux", order = [l,k]))
    #         k += 1
    #     elif pulse.type is PulseType.BIAS:
    #         Zhsequence.append(self.select_pulse(pulse, "Bias", order = [l,m]))
    #         m += 1
    #     elif pulse.type is PulseType.FLUX_COUPLER:
    #         Zhsequence.append(self.select_pulse(pulse, "Flux_Coupler", order = [l,r]))
    #         r += 1
    #     l += 1

    def def_calibration(self):
        self.calib = lo.Calibration()

        for it in range(len(self.qubits)):
            qubit = self.qubits[it]

            self.calib[f"/logical_signal_groups/q{qubit}/measure_line"] = lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=self.instruments["shfqc_qa"]["settings"][f"if_frequency_{qubit}"],
                    # modulation_type=lo.ModulationType.HARDWARE,
                    modulation_type=lo.ModulationType.SOFTWARE,
                ),
                local_oscillator=lo.Oscillator(
                    frequency=self.instruments["shfqc_qa"]["settings"]["lo_frequency"],
                ),
                range=self.instruments["shfqc_qa"]["settings"]["output_range"],
                # port_delay=self.settings["readout_delay"],
            )
            self.calib[f"/logical_signal_groups/q{qubit}/acquire_line"] = lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=self.instruments["shfqc_qa"]["settings"][f"if_frequency_{qubit}"],
                    modulation_type=lo.ModulationType.SOFTWARE,
                ),
                local_oscillator=lo.Oscillator(
                    frequency=self.instruments["shfqc_qa"]["settings"]["lo_frequency"],
                ),
                range=self.instruments["shfqc_qa"]["settings"]["input_range"],
                port_delay=10e-9,  # applied to corresponding instrument node, bound to hardware limits
                # port_delay=self.settings["readout_delay"],
            )

            self.calib[f"/logical_signal_groups/q{qubit}/flux_line"] = lo.SignalCalibration(
                # modulation_type=lo.ModulationType.HARDWARE,
                range=self.instruments["hdawg"]["settings"]["flux_range"],
                port_delay=0,  # applied to corresponding instrument node, bound to hardware limits
                delay_signal=0,
                voltage_offset=[0.02, 0.01],
            )

            self.calib[f"/logical_signal_groups/q{qubit}/drive_line"] = lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=self.instruments["shfqc_qc"]["settings"][f"if_frequency_{qubit}"],
                    modulation_type=lo.ModulationType.HARDWARE,
                ),
                local_oscillator=lo.Oscillator(
                    frequency=self.instruments["shfqc_qc"]["settings"]["lo_frequency"],
                ),
                range=self.instruments["shfqc_qc"]["settings"]["drive_range"],
            )

    # Set channel map
    def set_maps(self):
        self.map_q = {}

        for qubit in self.addressed_qubits:
            if any(self.sequence_drive):
                if any(self.sequence_flux):
                    self.map_q[f"drive{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals[
                        "drive_line"
                    ]
                    self.map_q[f"flux{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals[
                        "flux_line"
                    ]
                    self.map_q[f"measure{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals[
                        "measure_line"
                    ]
                    self.map_q[f"acquire{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals[
                        "acquire_line"
                    ]
                else:
                    self.map_q[f"drive{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals[
                        "drive_line"
                    ]
                    self.map_q[f"measure{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals[
                        "measure_line"
                    ]
                    self.map_q[f"acquire{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals[
                        "acquire_line"
                    ]

            elif any(self.sequence_flux):
                self.map_q[f"flux{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["flux_line"]
                self.map_q[f"measure{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals[
                    "measure_line"
                ]
                self.map_q[f"acquire{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals[
                    "acquire_line"
                ]

            else:
                self.map_q[f"measure{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals[
                    "measure_line"
                ]
                self.map_q[f"acquire{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals[
                    "acquire_line"
                ]

    # Reload settings together if possible
    def reload_settings(self):
        with open(self.runcard_file) as file:
            self.settings = yaml.safe_load(file)
        if self.is_connected:
            self.setup(**self.settings)

    def apply_settings(self):
        self.def_calibration()
        self.Zsetup.set_calibration(self.calib)

    def sequences_to_ZurichPulses(self, sequences, sweepers=None):
        self.sequences = sequences

        sequence_Z_drives = []
        sequence_Z_readouts = []
        sequence_Z_weights = []
        sequence_Z_fluxs = []
        Delays = []
        rel_phases = []
        Drive_durations = []
        addressed_qubits = []

        Starts = []
        Durations = []

        for l in range(len(sequences)):
            sequence = sequences[l]

            sequence_Z_drive = []
            sequence_Z_readout = []
            sequence_Z_weight = []
            sequence_Z_flux = []
            starts = []
            durations = []
            rel_phase = []

            i = 0
            j = 0
            k = 0
            Drive_duration = 0

            for pulse in sequence:
                starts.append(pulse.start)
                durations.append(pulse.duration)
                rel_phase.append(pulse.relative_phase)

                qubit = pulse.qubit
                if qubit in addressed_qubits:
                    pass
                else:
                    addressed_qubits.append(qubit)

                if str(pulse.type) == "PulseType.DRIVE":
                    if str(pulse.shape) == "Rectangular()":
                        sequence_Z_drive.append(
                            lo.pulse_library.const(
                                uid=(f"drive_{qubit}_" + str(l) + "_" + str(i)),
                                # length=pulse.duration * 1e-9,
                                length=round(pulse.duration * 1e-9, 9),
                                amplitude=pulse.amplitude,
                            )
                        )
                    elif "Gaussian" in str(pulse.shape):
                        sigma = str(pulse.shape).removeprefix("Gaussian(")
                        sigma = float(sigma.removesuffix(")"))
                        sequence_Z_drive.append(
                            lo.pulse_library.gaussian(
                                uid=(f"drive_{qubit}_" + str(l) + "_" + str(i)),
                                # length=pulse.duration * 1e-9,
                                length=round(pulse.duration * 1e-9, 9),
                                amplitude=pulse.amplitude,
                                sigma=2 / sigma,
                            )
                        )
                    elif "Drag" in str(pulse.shape):
                        params = str(pulse.shape).removeprefix("Drag(")
                        params = params.removesuffix(")")
                        params = params.split(",")
                        sigma = float(params[0])
                        beta = float(params[1])
                        sequence_Z_drive.append(
                            lo.pulse_library.drag(
                                uid=(f"drive_{qubit}_" + str(l) + "_" + str(i)),
                                # length=pulse.duration * 1e-9,
                                length=round(pulse.duration * 1e-9, 9),
                                amplitude=pulse.amplitude,
                                sigma=2 / sigma,
                                beta=beta,
                                # beta=2 / beta,
                            )
                        )

                    i += 1
                if str(pulse.type) == "PulseType.READOUT":
                    if str(pulse.shape) == "Rectangular()":
                        sequence_Z_readout.append(
                            lo.pulse_library.const(
                                uid=(f"readout_{qubit}_" + str(l) + "_" + str(j)),
                                # length=pulse.duration * 1e-9,
                                length=round(pulse.duration * 1e-9, 9),
                                amplitude=pulse.amplitude,
                            )
                        )

                        sequence_Z_weight.append(
                            lo.pulse_library.const(
                                uid="readout_weighting_function" + str(l) + "_" + str(j),
                                # length=pulse.duration * 1e-9,
                                length=round(pulse.duration * 1e-9, 9),
                                amplitude=1.0,
                            )
                        )
                    j += 1

                if str(pulse.type) == "PulseType.FLUX":
                    if str(pulse.shape) == "Rectangular()":
                        sequence_Z_flux.append(
                            lo.pulse_library.const(
                                uid=(f"flux_{qubit}_" + str(l) + "_" + str(k)),
                                # length=pulse.duration * 1e-9,
                                length=round(pulse.duration * 1e-9, 9),
                                amplitude=pulse.amplitude,
                            )
                        )
                    k += 1

            delays = []
            for i in range(len(starts) - 1):
                delays.append(starts[i + 1] - durations[i])

            Drive_durations.append(Drive_duration)
            sequence_Z_fluxs.append(sequence_Z_flux)
            sequence_Z_readouts.append(sequence_Z_readout)
            sequence_Z_weights.append(sequence_Z_weight)
            sequence_Z_drives.append(sequence_Z_drive)
            Delays.append(delays)
            rel_phases.append(rel_phase)

            Starts.append(starts)
            Durations.append(durations)

        self.delays = Delays
        self.sequence_drive = sequence_Z_drives
        self.sequence_readout = sequence_Z_readouts
        self.sequence_flux = sequence_Z_fluxs
        self.sequence_weight = sequence_Z_weights
        self.rel_phases = rel_phases
        self.Drive_durations = Drive_durations
        self.addressed_qubits = addressed_qubits

        self.starts = Starts
        self.durations = Durations

        self.sweepers = sweepers
        if sweepers != None:
            self.sweepers = sweepers
            sweepers_Zh = []
            for sweep in sweepers:
                sweepers_Zh.append(
                    lo.LinearSweepParameter(
                        uid=sweep.parameter.name,
                        start=sweep.values[0],
                        stop=sweep.values[-1],
                        count=sweep.values.shape[0],
                    )
                )
            self.sweepers_Zh = sweepers_Zh

    # TODO: 2 consecutive experiments without compiling (Needs memory fix)
    # TODO: Multiplex (For readout)
    # TODO: All the posible sweeps
    # TODO: Select averaging and acquisition modes
    def create_exp(self):
        signals = []
        if any(self.sequence_drive):
            if any(self.sequence_flux):
                for qubit in self.addressed_qubits:
                    signals.append(lo.ExperimentSignal(f"drive{qubit}"))
                    signals.append(lo.ExperimentSignal(f"flux{qubit}"))
                    signals.append(lo.ExperimentSignal(f"measure{qubit}"))
                    signals.append(lo.ExperimentSignal(f"acquire{qubit}"))
            else:
                for qubit in self.addressed_qubits:
                    signals.append(lo.ExperimentSignal(f"drive{qubit}"))
                    signals.append(lo.ExperimentSignal(f"measure{qubit}"))
                    signals.append(lo.ExperimentSignal(f"acquire{qubit}"))
        elif any(self.sequence_flux):
            for qubit in self.addressed_qubits:
                signals.append(lo.ExperimentSignal(f"flux{qubit}"))
                signals.append(lo.ExperimentSignal(f"measure{qubit}"))
                signals.append(lo.ExperimentSignal(f"acquire{qubit}"))
        else:
            for qubit in self.addressed_qubits:
                signals.append(lo.ExperimentSignal(f"measure{qubit}"))
                signals.append(lo.ExperimentSignal(f"acquire{qubit}"))

        exp = lo.Experiment(
            uid="Sequence",
            signals=signals,
        )

        # for j in range(len(self.sequences)):
        #     self.iteration = j

        # For the Resonator Spec
        with exp.acquire_loop_rt(
            uid="shots",
            count=self.settings["hardware_avg"],
            # repetition_mode= lo.RepetitionMode.CONSTANT,
            # repetition_time= 20e-6,
            acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
            # acquisition_type=lo.AcquisitionType.INTEGRATION,
            averaging_mode=lo.AveragingMode.CYCLIC,
            # averaging_mode=lo.AveragingMode.SINGLE_SHOT,
        ):
            # For multiplex readout
            # with exp.acquire_loop_rt(
            #     uid="shots",
            #     count=self.settings["hardware_avg"],
            #     # repetition_mode= lo.RepetitionMode.CONSTANT,
            #     # repetition_time= 20e-6,
            #     acquisition_type=lo.AcquisitionType.INTEGRATION,
            #     averaging_mode=lo.AveragingMode.CYCLIC,
            #     # averaging_mode=lo.AveragingMode.SINGLE_SHOT,
            # ):

            if self.sweepers is not None:
                if len(self.sweepers) == 1:
                    if self.sweepers[0].parameter.name == "frequency":
                        with exp.sweep(parameter=self.sweepers_Zh[0]):
                            k = 0

                            self.select_exp(exp)

                            qubit = self.sweepers[0].pulses[0].qubit
                        # define experiment calibration - sweep over qubit drive frequency
                        exp_calib = lo.Calibration()
                        # exp_calib[f"{line}{qubit}"] = lo.SignalCalibration(
                        exp_calib[f"measure{qubit}"] = lo.SignalCalibration(
                            oscillator=lo.Oscillator(
                                frequency=self.sweepers_Zh[0],
                                modulation_type=lo.ModulationType.HARDWARE,
                            )
                        )

                        exp.set_calibration(exp_calib)

                elif len(self.sweepers) == 2:
                    # for sweep in self.sweepers:

                    # if sweep.parameter == "freq":

                    self.amplitude = self.sweepers_Zh[1]
                    with exp.sweep(
                        uid="sweep_freq", parameter=self.sweepers_Zh[0], alignment=lo.SectionAlignment.RIGHT
                    ):
                        with exp.sweep(uid="sweep_param", parameter=self.sweepers_Zh[1]):
                            k = 0

                            self.select_exp(exp)

                    qubit = self.sweepers[0].pulses[0].qubit
                    # define experiment calibration - sweep over qubit drive frequency
                    exp_calib = lo.Calibration()
                    exp_calib[f"measure{qubit}"] = lo.SignalCalibration(
                        oscillator=lo.Oscillator(
                            frequency=self.sweepers_Zh[0],
                            modulation_type=lo.ModulationType.HARDWARE,
                        )
                    )

                    exp.set_calibration(exp_calib)

                elif len(self.sweepers) == 5:
                    # for sweep in self.sweepers:
                    if self.sweepers[0].parameter == "freq_qs_0":
                        with exp.sweep(
                            uid="sweep_freq", parameter=self.sweepers_Zh, alignment=lo.SectionAlignment.RIGHT
                        ):
                            k = 0
                            self.select_exp(exp)

                        # qubit = self.sweepers[0].pulses[0].qubit
                        exp_calib = lo.Calibration()
                        for qubit in self.addressed_qubits:
                            # define experiment calibration - sweep over qubit drive frequency
                            exp_calib[f"drive{qubit}"] = lo.SignalCalibration(
                                oscillator=lo.Oscillator(
                                    frequency=self.sweepers_Zh[qubit],
                                    modulation_type=lo.ModulationType.HARDWARE,
                                )
                            )

                        exp.set_calibration(exp_calib)
                    elif self.sweepers[0].parameter == "rabi_lenght_0" or self.sweepers[0].parameter == "rabi_amp_0":
                        with exp.sweep(uid="sweep", parameter=self.sweepers_Zh, alignment=lo.SectionAlignment.RIGHT):
                            self.select_exp(exp)

            else:
                self.amplitude = 0.1
                self.select_exp(exp)

        self.set_maps()
        exp.set_signal_map(self.map_q)

        self.experiment = exp

    def select_exp(self, exp):
        if any(self.sequence_drive):
            if any(self.sequence_flux):
                for j in range(len(self.sequences)):
                    self.iteration = j
                    self.Flux(exp)
                    self.Drive(exp)
                    self.Measure(exp)
                    self.qubit_reset(exp)

            else:
                for j in range(len(self.sequences)):
                    self.iteration = j
                    self.Drive(exp)
                    # self.Drive_Rabi(exp)
                    self.Measure(exp)
                    self.qubit_reset(exp)

        elif any(self.sequence_flux):
            for j in range(len(self.sequences)):
                self.iteration = j
                self.Flux(exp)
                self.Measure(exp)
                self.qubit_reset(exp)

        else:
            for j in range(len(self.sequences)):
                self.iteration = j
                self.Measure(exp)
                self.qubit_reset(exp)

    # Flux on all qubits
    def Flux(self, exp):
        j = self.iteration
        with exp.section(uid=f"sequence{j}_flux_bias", alignment=lo.SectionAlignment.RIGHT):
            for pulse in self.sequence_flux[j]:
                qubit = pulse.uid.split("_")[1]
                exp.play(signal=f"flux{qubit}", pulse=pulse, amplitude=self.amplitude)
            for qubit in self.addressed_qubits:
                exp.delay(signal=f"flux{qubit}", time=self.settings["readout_delay"])

    def Drive(self, exp):
        j = self.iteration
        with exp.section(uid=f"sequence{j}_drive", alignment=lo.SectionAlignment.RIGHT):
            i = 0
            # for qubit in self.addressed_qubits:
            # exp.delay(signal=f"drive{qubit}", time = 10e-9) #ramp up
            for pulse in self.sequence_drive[j]:
                qubit = pulse.uid.split("_")[1]
                exp.play(signal=f"drive{qubit}", pulse=pulse, phase=self.rel_phases[j][i])
                exp.reserve(signal=f"measure{qubit}")
                if self.delays[j][i] > 0:
                    qubit = pulse.uid.split("_")[1]
                    exp.delay(signal=f"drive{qubit}", time=self.delays[j][i] * 1e-9)
                i += 1

    def Drive_Rabi(self, exp):
        j = self.iteration

        with exp.section(uid=f"sequence{j}_drive", alignment=lo.SectionAlignment.RIGHT):
            i = 0
            # for qubit in self.addressed_qubits:
            # exp.delay(signal=f"drive{qubit}", time = 10e-9) #ramp up
            for pulse in self.sequence_drive[j]:
                qubit = pulse.uid.split("_")[1]

                if self.sweepers[0].parameter == "rabi_lenght_0":
                    length = self.sweepers_Zh[int(qubit)]
                    amp = pulse.amplitude
                else:
                    amp = self.sweepers_Zh[int(qubit)]
                    length = pulse.length

                exp.play(signal=f"drive{qubit}", pulse=pulse, phase=self.rel_phases[j][i], length=length, amplitude=amp)
                exp.reserve(signal=f"measure{qubit}")
                if self.delays[j][i] > 0:
                    qubit = pulse.uid.split("_")[1]
                    exp.delay(signal=f"drive{qubit}", time=self.delays[j][i] * 1e-9)
                i += 1

    def Measure(self, exp):
        # qubit readout pulse and data acquisition
        j = self.iteration
        with exp.section(uid=f"sequence{j}_measure"):
            # with exp.section(uid=f"sequence{j}_measure", play_after=f"sequence{j}_drive"):
            # exp.reserve("drive")
            i = 0
            for pulse in self.sequence_readout[j]:
                qubit = pulse.uid.split("_")[1]
                # FIXME: Lenght
                exp.play(signal=f"measure{qubit}", pulse=pulse)
                # exp.play(signal=f"measure{qubit}", pulse=pulse, phase=self.rel_phases[j][i], lenght = 1e-6)
                # FIXME: Handle
                if self.sweepers != None:
                    if len(self.sweepers) == 2:
                        exp.acquire(signal=f"acquire{qubit}", handle=f"sequence", kernel=self.sequence_weight[j][i])
                    elif len(self.sweepers) == 5:
                        exp.acquire(
                            signal=f"acquire{qubit}", handle=f"sequence_{i}_{j}", kernel=self.sequence_weight[j][i]
                        )
                    elif len(self.sweepers) == 1:
                        exp.acquire(signal=f"acquire{qubit}", handle=f"sequence_{j}", kernel=self.sequence_weight[j][i])

                elif len(self.sequence_readout[j]) > 1:
                    exp.acquire(signal=f"acquire{qubit}", handle=f"sequence_{i}_{j}", kernel=self.sequence_weight[j][i])
                else:
                    exp.acquire(signal=f"acquire{qubit}", handle=f"sequence_{j}", kernel=self.sequence_weight[j][i])
                i += 1

    def qubit_reset(self, exp):
        j = self.iteration
        # relax time or fast reset after readout - for signal processing and qubit relaxation to ground state
        if self.settings["Fast_reset"] == True:
            for qubit in self.addressed_qubits:
                with exp.section(uid=f"fast_reset", play_after=f"sequence{j}_measure"):
                    with exp.match_local(handle=f"acquire{qubit}"):
                        with exp.case(state=0):
                            pass
                            # exp.play(some_pulse)
                        with exp.case(state=1):
                            pass
                            # exp.play(some_other_pulse)
        else:
            with exp.section(uid=f"relax_{j}", play_after=f"sequence{j}_measure"):
                for qubit in self.addressed_qubits:
                    exp.delay(signal=f"measure{qubit}", time=self.settings["readout_delay"])

    def sequence_to_Zurichpulses(self, sequence):
        self.sequence = sequence
        sequence_Z_drive = []
        sequence_Z_readout = []
        starts = []
        durations = []
        self.rel_phases = []
        i = 0
        j = 0
        for pulse in sequence:
            starts.append(pulse.start)
            durations.append(pulse.duration)
            # self.rel_phases.append(pulse.relative_phase)

            if str(pulse.type) == "PulseType.DRIVE":
                if str(pulse.shape) == "Rectangular()":
                    sequence_Z_drive.append(
                        lo.pulse_library.const(
                            uid=("drive" + str(i)),
                            length=pulse.duration * 1e-9,
                            amplitude=pulse.amplitude,
                        )
                    )
                elif "Gaussian" in str(pulse.shape):
                    sigma = str(pulse.shape).removeprefix("Gaussian(")
                    sigma = float(sigma.removesuffix(")"))
                    sequence_Z_drive.append(
                        lo.pulse_library.gaussian(
                            uid=("drive" + str(i)),
                            length=pulse.duration * 1e-9,
                            amplitude=pulse.amplitude,
                            sigma=2 / sigma,
                        )
                    )
                elif "Drag" in str(pulse.shape):
                    params = str(pulse.shape).removeprefix("Drag(")
                    params = params.removesuffix(")")
                    params = params.split(",")
                    sigma = float(params[0])
                    beta = float(params[1])
                    sequence_Z_drive.append(
                        lo.pulse_library.drag(
                            uid=("drive" + str(i)),
                            length=pulse.duration * 1e-9,
                            amplitude=pulse.amplitude,
                            sigma=2 / sigma,
                            beta=beta,
                            # beta=2 / beta,
                        )
                    )

            i += 1
            if str(pulse.type) == "PulseType.READOUT":
                if str(pulse.shape) == "Rectangular()":
                    sequence_Z_readout.append(
                        lo.pulse_library.const(
                            uid=("readout" + str(j)),
                            length=pulse.duration * 1e-9,
                            amplitude=pulse.amplitude,
                        )
                    )

                    self.readout_weighting_function = lo.pulse_library.const(
                        uid="readout_weighting_function",
                        length=2 * pulse.duration * 1e-9,
                        amplitude=1.0,
                    )
            j += 1

        delays = []
        for i in range(len(starts) - 1):
            delays.append(starts[i + 1] - durations[i])

        self.delays = delays
        self.sequence_drive = sequence_Z_drive
        self.sequence_readout = sequence_Z_readout

    # Separe play and sweep and add relax and shots.
    # And list of Qubits(Object)
    def execute_sequences(self, sequences, sweepers=None):
        # if self.sequence == sequence:
        #     self.repeat_seq()
        # else:
        #     self.sequence_to_ZurichPulses(sequence)
        #     self.sequencePulses_to_exp()
        #     self.run_seq()

        self.sequences_to_ZurichPulses(sequences, sweepers)
        self.create_exp()
        self.run_seq()

        spec_res = []
        msr = []
        phase = []
        i = []
        q = []

        if self.sweepers != None:
            if len(self.sweepers) == 2:
                for j in range(self.sweepers[0].count):
                    for k in range(self.sweepers[1].count):
                        datapoint = self.results.get_data("sequence")[j][k]
                        msr.append(abs(datapoint))
                        phase.append(np.angle(datapoint))
                        i.append(datapoint.real)
                        q.append(datapoint.imag)

                return msr, phase, i, q

            elif len(self.sweepers) == 5:
                for j in range(5):
                    spec_res.append(self.results.get_data(f"sequence_{j}_{0}"))
                    msr.append(abs(spec_res[j]))
                    phase.append(np.angle(spec_res[j]))
                    i.append(spec_res[j].real)
                    q.append(spec_res[j].imag)

                return msr, phase, i, q

            elif len(self.sweepers) == 1:
                # handles = result.result_handles
                # results = {}
                # for pulse in ro_pulses:
                #     serial = pulse.serial
                #     ires = handles.get(f"{serial}_I").fetch_all()
                #     qres = handles.get(f"{serial}_Q").fetch_all()
                #     if f"{serial}_shots" in handles:
                #         shots = handles.get(f"{serial}_shots").fetch_all().astype(int)
                #     else:
                #         shots = None
                #     results[pulse.qubit] = results[serial] = ExecutionResults.from_components(ires, qres, shots)
                # return results

                results = {}
                for j in range(len(self.sequences)):
                    for pulse in sequences[j].ro_pulses:
                        spec_res = self.results.get_data(f"sequence_{j}")
                        i = spec_res.real
                        q = spec_res.imag

                        # spec_res.append(self.results.get_data(f"sequence_{j}"))
                        # msr.append(abs(spec_res[j]))
                        # phase.append(np.angle(spec_res[j]))
                        # i.append(spec_res[j].real)
                        # q.append(spec_res[j].imag)

                shots = self.settings["hardware_avg"]
                results[pulse.qubit] = ExecutionResults.from_components(i, q, shots)

                return results

        elif len(self.sequence_readout[0]) > 1:
            for j in range(len(self.sequences)):
                for k in range(len(self.sequence_readout[j])):
                    datapoint = self.results.get_data(f"sequence_{k}_{j}")
                    msr.append(abs(datapoint))
                    phase.append(np.angle(datapoint))
                    i.append(datapoint.real)
                    q.append(datapoint.imag)

            return msr, phase, i, q

        else:
            for j in range(len(self.sequences)):
                spec_res.append(self.results.get_data(f"sequence_{j}"))
                msr.append(abs(spec_res[j]))
                phase.append(np.angle(spec_res[j]))
                i.append(spec_res[j].real)
                q.append(spec_res[j].imag)

            return msr, phase, i, q

    def sequencepulses_to_exp(self):
        # Create Experiment

        if len(self.sequence_drive) != 0:
            exp = lo.Experiment(
                uid="Sequence",
                signals=[
                    lo.ExperimentSignal("drive"),
                    lo.ExperimentSignal("measure"),
                    lo.ExperimentSignal("acquire"),
                ],
            )

            ## experimental pulse sequence
            # outer loop - real-time, cyclic averaging in standard integration mode
            with exp.acquire_loop_rt(
                uid="shots",
                count=self.settings["hardware_avg"],
                averaging_mode=lo.AveragingMode.SEQUENTIAL,
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                # averaging_mode=lo.AveragingMode.CYCLIC,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):
                # # inner loop - real-time sweep of qubit drive pulse amplitude
                # with exp.sweep(uid="sweep", parameter=sweep_rel_flat, alignment=SectionAlignment.RIGHT):
                # qubit excitation - pulse amplitude will be swept
                with exp.section(uid="qubit_excitation", alignment=lo.SectionAlignment.RIGHT):
                    i = 0
                    for pulse in self.sequence_drive:
                        exp.play(signal="drive", pulse=pulse, phase=self.rel_phases[i])

                        if self.delays[i] > 0:
                            exp.delay(signal="drive", time=self.delays[i] * 1e-9)
                        i += 1

                # qubit readout pulse and data acquisition

                with exp.section(uid="qubit_readout"):
                    for pulse in self.sequence_readout:
                        exp.reserve(signal="drive")

                        exp.play(signal="measure", pulse=pulse, phase=self.rel_phases[i])

                        integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]

                        exp.acquire(signal="acquire", handle="sequence", length=integration_time)

                        # exp.acquire(
                        #     signal="acquire",
                        #     handle="Sequence",
                        #     kernel=self.readout_weighting_function,
                        # )

                # relax time after readout - for signal processing and qubit relaxation to ground state
                with exp.section(uid="relax"):
                    exp.delay(signal="measure", time=self.settings["readout_delay"])

        # TODO: Add features of above to else
        else:
            exp = lo.Experiment(
                uid="Sequence",
                signals=[
                    lo.ExperimentSignal("measure0"),
                    lo.ExperimentSignal("acquire0"),
                ],
            )
            ## experimental pulse sequence
            # outer loop - real-time, cyclic averaging in standard integration mode
            with exp.acquire_loop_rt(
                uid="shots",
                count=self.settings["hardware_avg"],
                averaging_mode=lo.AveragingMode.CYCLIC,
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):
                # # inner loop - real-time sweep of qubit drive pulse amplitude
                # qubit readout pulse and data acquisition

                i = 0
                with exp.section(uid="qubit_readout"):
                    for pulse in self.sequence_readout:
                        exp.play(signal="measure0", pulse=pulse)

                        integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]

                        exp.acquire(signal="acquire0", handle="sequence", length=integration_time)

                        # exp.acquire(
                        #     signal="acquire",
                        #     handle="Sequence",
                        #     kernel=self.readout_weighting_function,
                        # )

                # relax time after readout - for signal processing and qubit relaxation to ground state
                with exp.section(uid="relax"):
                    exp.delay(signal="measure0", time=self.settings["readout_delay"])

        qubit = 0
        map_q = {}
        map_q[f"measure{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["measure_line"]
        map_q[f"acquire{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["acquire_line"]

        exp.set_signal_map(map_q)

        self.experiment = exp

    def execute_pulse_sequence(self, sequence):
        self.sequence_to_Zurichpulses(sequence)
        self.sequencepulses_to_exp()
        self.run_seq()

        spec_res = self.results.get_data("sequence")

        msr = abs(spec_res)
        # phase = np.unwrap(np.angle(spec_res))
        phase = np.angle(spec_res)
        i = spec_res.real
        q = spec_res.imag

        return msr, phase, i, q

    def compile_exp(self, exp):
        self.exp = self.session.compile(exp)

    def run_exp(self):
        self.results = self.session.run(self.exp)

    def run_seq(self):
        # compiler_settings = {
        #     "SHFSG_FORCE_COMMAND_TABLE": True,
        #     "SHFSG_MIN_PLAYWAVE_HINT": 32,
        #     "SHFSG_MIN_PLAYZERO_HINT": 32,
        # }

        # self.exp = self.session.compile(self.experiment, compiler_settings=compiler_settings)

        self.exp = self.session.compile(self.experiment)
        self.results = self.session.run(self.exp, self.emulation)

    def run_multi(self):
        compiler_settings = {
            "SHFSG_FORCE_COMMAND_TABLE": True,
            "SHFSG_MIN_PLAYWAVE_HINT": 32,
            "SHFSG_MIN_PLAYZERO_HINT": 32,
        }

        self.exp = self.session.compile(self.experiment, compiler_settings=compiler_settings)
        self.results = self.session.run(self.exp, self.emulation)

    def repeat_seq(self):
        self.results = self.session.run(do_simulation=self.emulation)
