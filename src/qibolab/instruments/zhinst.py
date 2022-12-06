# -*- coding: utf-8 -*-
import laboneq.simple as lo
import matplotlib.pyplot as plt
import numpy as np
import yaml

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.pulses import PulseSequence


class SHFQC_QA(AbstractInstrument):
    def __init__(self, name, address, runcard, use_emulation):
        super().__init__(name, address)
        self.device = None

        self.channels = None
        self.lo_frequency = None  # in units of Hz
        self.gain = None
        self.input_range = None  # in units of dBm
        self.output_range = None  # in units of dBm

        self._latest_sequence = None
        self.descriptor_path = "/home/admin/Juan/qibolab/src/qibolab/runcards/descriptor_shfqc.yml"
        self.runcard_file = runcard
        self.emulation = use_emulation

        with open(runcard, "r") as file:
            settings = yaml.safe_load(file)

        self.setup(**settings)
        self.def_calibration()
        self.Z_setup()
        self.connect(use_emulation=self.emulation)

    def compile_exp(self, exp):
        self.exp = self.session.compile(exp)

    def run_exp(self):
        self.results = self.session.run(self.exp)

    def run_seq(self):
        self.exp = self.session.compile(self.experiment)
        self.results = self.session.run(self.exp, self.emulation)
        # self.results = self.session.run(self.experiment, self.emulation)

    def repeat_seq(self):
        print("repeat")
        self.results = self.session.run(do_simulation=self.emulation)
        # self.results = self.session.run(self.experiment, self.emulation)

    # TODO: Proper calibration and channel map
    def def_calibration(self):
        self.calib = lo.Calibration()

        for it in range(len(self.qubits)):
            qubit = self.qubits[it]

            self.calib[f"/logical_signal_groups/q{qubit}/drive_line"] = lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=self.instruments["shfqc_qc"]["settings"]["if_frequency"],
                    modulation_type=lo.ModulationType.HARDWARE,
                ),
                local_oscillator=lo.Oscillator(
                    frequency=self.instruments["shfqc_qc"]["settings"]["lo_frequency"],
                ),
                range=self.instruments["shfqc_qc"]["settings"]["drive_range"],
            )
            self.calib[f"/logical_signal_groups/q{qubit}/measure_line"] = lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=self.instruments["shfqc_qa"]["settings"]["if_frequency"],
                    modulation_type=lo.ModulationType.HARDWARE,
                ),
                local_oscillator=lo.Oscillator(
                    frequency=self.instruments["shfqc_qa"]["settings"]["lo_frequency"],
                ),
                range=self.instruments["shfqc_qa"]["settings"]["output_range"],
                # port_delay=self.settings["readout_delay"],
            )
            self.calib[f"/logical_signal_groups/q{qubit}/acquire_line"] = lo.SignalCalibration(
                # oscillator=lo.Oscillator(
                #     frequency=self.instruments["shfqc_qa"]["settings"]["if_frequency"],
                #     modulation_type=lo.ModulationType.SOFTWARE,
                # ),
                local_oscillator=lo.Oscillator(
                    frequency=self.instruments["shfqc_qa"]["settings"]["lo_frequency"],
                ),
                range=self.instruments["shfqc_qa"]["settings"]["input_range"],
                port_delay=10e-9,  # applied to corresponding instrument node, bound to hardware limits
                # port_delay=self.settings["readout_delay"],
            )

    def set_map(self):
        if len(self.sequence_drive) != 0:
            self.map_q0 = {
                "drive": self.Zsetup.logical_signal_groups["q0"].logical_signals["drive_line"],
                "measure": self.Zsetup.logical_signal_groups["q0"].logical_signals["measure_line"],
                "acquire": self.Zsetup.logical_signal_groups["q0"].logical_signals["acquire_line"],
            }
        else:
            self.map_q0 = {
                "measure": self.Zsetup.logical_signal_groups["q0"].logical_signals["measure_line"],
                "acquire": self.Zsetup.logical_signal_groups["q0"].logical_signals["acquire_line"],
            }

    def connect(self, use_emulation):
        global cluster
        if not self.is_connected:
            for attempt in range(3):
                try:
                    self.session = lo.Session(self.Zsetup)
                    # self.device = self.session.connect(self.address)
                    self.device = self.session.connect(do_emulation=use_emulation)
                    # self.device.reset()
                    cluster = self.device
                    self.is_connected = True
                    break
                except Exception as exc:
                    print(f"Unable to connect:\n{str(exc)}\nRetrying...")
            if not self.is_connected:
                raise InstrumentException(self, f"Unable to connect to {self.name}")

    # TODO: Join setup and Z_setup
    def setup(self, **kwargs):
        self.resonator_type = kwargs.get("resonator_type")
        self.qubits = kwargs.get("qubits")
        self.channels = kwargs.get("channels")
        self.gain = kwargs.get("gain")
        self.lo_frequency = kwargs.get("lo_frequency")
        self.input_range = kwargs.get("input_range")
        self.output_range = kwargs.get("output_range")
        self.characterization = kwargs.get("characterization")
        self.instruments = kwargs.get("instruments")
        self.native_gates = kwargs.get("native_gates")
        self.settings = kwargs.get("settings")
        self.descriptor = kwargs.get("descriptor")
        self.qubit_channel_map = kwargs.get("qubit_channel_map")
        self.descriptor = kwargs.get("instrument_list")
        self.parameters = kwargs.get("parameters")
        self.sequence = PulseSequence()

    def Z_setup(self):
        # self.Zsetup = lo.DeviceSetup.from_descriptor(
        #     yaml_text = self.descriptor,
        #     server_host="localhost",
        #     server_port="8004",
        #     setup_name=self.name,
        # )

        self.Zsetup = lo.DeviceSetup.from_yaml(
            filepath=self.descriptor_path,
            server_host="localhost",
            server_port="8004",
            setup_name=self.name,
        )

        self.Zsetup.set_calibration(self.calib)

    # TODO: Check reload settings properly working
    def reload_settings(self):
        with open(self.runcard_file, "r") as file:
            self.settings = yaml.safe_load(file)

        if self.is_connected:
            self.setup(**self.settings)

        # self.hardware_avg = self.settings["settings"]["hardware_avg"]
        # self.sampling_rate = self.settings["settings"]["sampling_rate"]
        # self.repetition_duration = self.settings["settings"]["repetition_duration"]

        # # Load Characterization settings
        # self.characterization = self.settings["characterization"]
        # # Load Native Gates
        # self.native_gates = self.settings["native_gates"]

    def apply_settings(self):
        self.def_calibration()
        self.Zsetup.set_calibration(self.calib)

    # FIXME: What are these for ???
    def start(self):
        pass

    def stop(self):
        # TODO: Remember to stop sequencer here
        pass

    def disconnect(self):
        if self.is_connected:
            self.session = Session("localhost")
            self.device = self.session.disconnect_device(self.address)
            self.is_connected = False
            global cluster
            cluster = None
        else:
            print(f"Already disconnected")

    def sequence_to_ZurichPulses(self, sequence):
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
            self.rel_phases.append(pulse.relative_phase)

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

    def sequences_to_ZurichPulses(self, sequences):

        self.sequences = sequences
        sequence_Z_drives = []
        sequence_Z_readouts = []
        Delays = []
        rel_phases = []
        Drive_durations = []

        for k in range(len(sequences)):
            sequence = sequences[k]

            sequence_Z_drive = []
            sequence_Z_readout = []
            starts = []
            durations = []
            rel_phase = []
            i = 0
            j = 0
            Drive_duration = 0

            for pulse in sequence:

                starts.append(pulse.start)
                durations.append(pulse.duration)
                rel_phase.append(pulse.relative_phase)

                if str(pulse.type) == "PulseType.DRIVE":

                    Drive_duration = (pulse.duration + pulse.start) * 1e-9

                    if str(pulse.shape) == "Rectangular()":
                        sequence_Z_drive.append(
                            lo.pulse_library.const(
                                uid=("drive_" + str(k) + "_" + str(i)),
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
                                uid=("readout_" + str(k) + "_" + str(j)),
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

            Drive_durations.append(Drive_duration)
            sequence_Z_readouts.append(sequence_Z_readout)
            sequence_Z_drives.append(sequence_Z_drive)
            Delays.append(delays)
            rel_phases.append(rel_phase)

        self.delays = Delays
        self.sequence_drive = sequence_Z_drives
        self.sequence_readout = sequence_Z_readouts
        self.rel_phases = rel_phases
        self.Drive_durations = Drive_durations

    def sequencesPulses_to_exp(self):
        # Create Experiment
        if len(self.sequence_drive[0]) != 0:
            exp = lo.Experiment(
                uid="Sequence",
                signals=[
                    lo.ExperimentSignal("drive"),
                    lo.ExperimentSignal("measure"),
                    lo.ExperimentSignal("acquire"),
                ],
            )

            self.map_q0 = {
                "drive": self.Zsetup.logical_signal_groups["q0"].logical_signals["drive_line"],
                "measure": self.Zsetup.logical_signal_groups["q0"].logical_signals["measure_line"],
                "acquire": self.Zsetup.logical_signal_groups["q0"].logical_signals["acquire_line"],
            }

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

                for j in range(len(self.sequence_readout)):
                    sequence_d = self.sequence_drive[j]
                    sequence_r = self.sequence_readout[j]

                    # # inner loop - real-time sweep of qubit drive pulse amplitude
                    # with exp.sweep(uid="sweep", parameter=sweep_rel_flat, alignment=SectionAlignment.RIGHT):
                    # qubit excitation - pulse amplitude will be swept
                    with exp.section(
                        uid="qubit_excitation", length=self.Drive_durations[j] * 10, alignment=lo.SectionAlignment.RIGHT
                    ):
                        i = 0
                        for pulse in sequence_d:
                            exp.play(signal="drive", pulse=pulse, phase=self.rel_phases[j][i])

                            if self.delays[j][i] > 0:
                                exp.delay(signal="drive", time=self.delays[j][i] * 1e-9)
                            i += 1

                    # qubit readout pulse and data acquisition
                    with exp.section(uid="qubit_readout"):
                        # for pulse in sequence_r:

                        exp.reserve(signal="drive")

                        # exp.play(signal="measure", pulse=pulse, phase =  self.rel_phases[j][i])
                        exp.play(signal="measure", pulse=sequence_r[0], phase=self.rel_phases[j][i])

                        integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]

                        exp.acquire(signal="acquire", handle="Sequence" + str(j), length=integration_time)

                        # exp.acquire(
                        #     signal="acquire",
                        #     handle="Sequence",
                        #     kernel=self.readout_weighting_function,
                        # )

                    # relax time after readout - for signal processing and qubit relaxation to ground state
                    with exp.section(uid="relax", length=self.Drive_durations[j] * 1000):
                        exp.delay(signal="measure", time=self.settings["readout_delay"])

        # TODO: Add features of above to else
        else:
            exp = lo.Experiment(
                uid="Sequence",
                signals=[
                    lo.ExperimentSignal("measure"),
                    lo.ExperimentSignal("acquire"),
                ],
            )

            self.map_q0 = {
                "measure": self.Zsetup.logical_signal_groups["q0"].logical_signals["measure_line"],
                "acquire": self.Zsetup.logical_signal_groups["q0"].logical_signals["acquire_line"],
            }

            ## experimental pulse sequence
            # outer loop - real-time, cyclic averaging in standard integration mode
            with exp.acquire_loop_rt(
                uid="shots",
                count=self.settings["hardware_avg"],
                # averaging_mode=lo.AveragingMode.CYCLIC,
                averaging_mode=lo.AveragingMode.SEQUENTIAL,
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):
                # # inner loop - real-time sweep of qubit drive pulse amplitude
                # qubit readout pulse and data acquisition

                # readout_weighting_function = lo.pulse_library.const(
                #     uid="readout_weighting_function",
                #     length=self.sequence_readout[0].duration * 1e-9,
                #     amplitude=1.0,
                # )

                for j in range(len(self.sequence_readout)):
                    sequence_r = self.sequence_readout[j]

                    with exp.section(length=5e-6, alignment=lo.SectionAlignment.RIGHT, uid="qubit_readout"):
                        # for pulse in self.sequence_readout:
                        # exp.play(signal="measure", pulse=pulse)
                        i = 0
                        exp.play(signal="measure", pulse=sequence_r[i], phase=self.rel_phases[j][i])

                        integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]

                        # exp.acquire(
                        #     signal="acquire",
                        #     handle=f"sequence{j}",
                        #     length=integration_time
                        # )

                        exp.acquire(
                            signal="acquire",
                            handle=f"sequence{j}",
                            kernel=self.readout_weighting_function,
                        )

                    # relax time after readout - for signal processing and qubit relaxation to ground state
                    with exp.section(uid="relax"):
                        exp.delay(signal="measure", time=self.settings["readout_delay"])

        # self.set_map()
        exp.set_signal_map(self.map_q0)
        print("hola")
        self.experiment = exp

    def sequencePulses_to_exp(self):
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

                        exp.acquire(signal="acquire", handle="Sequence", length=integration_time)

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
                    lo.ExperimentSignal("measure"),
                    lo.ExperimentSignal("acquire"),
                ],
            )
            ## experimental pulse sequence
            # outer loop - real-time, cyclic averaging in standard integration mode
            with exp.acquire_loop_rt(
                uid="shots",
                count=self.settings["hardware_avg"],
                # averaging_mode=lo.AveragingMode.CYCLIC,
                averaging_mode=lo.AveragingMode.SEQUENTIAL,
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):
                # # inner loop - real-time sweep of qubit drive pulse amplitude
                # qubit readout pulse and data acquisition

                # readout_weighting_function = lo.pulse_library.const(
                #     uid="readout_weighting_function",
                #     length=self.sequence_readout[0].duration * 1e-9,
                #     amplitude=1.0,
                # )

                with exp.section(uid="qubit_readout"):
                    for pulse in self.sequence_readout:
                        exp.play(signal="measure", pulse=pulse)

                        integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]

                        exp.acquire(signal="acquire", handle="Sequence", length=integration_time)

                        # exp.acquire(
                        #     signal="acquire",
                        #     handle="Sequence",
                        #     kernel=self.readout_weighting_function,
                        # )

                # relax time after readout - for signal processing and qubit relaxation to ground state
                with exp.section(uid="relax"):
                    exp.delay(signal="measure", time=self.settings["readout_delay"])

        self.set_map()
        exp.set_signal_map(self.map_q0)

        self.experiment = exp

    def execute_pulse_sequence_NoSamples(self, sequence):

        # if self.sequence == sequence:
        #     self.repeat_seq()
        # else:
        #     self.sequence_to_ZurichPulses(sequence)
        #     self.sequencePulses_to_exp()
        #     self.run_seq()

        self.sequence_to_ZurichPulses(sequence)
        self.sequencePulses_to_exp()
        self.run_seq()

        spec_res = self.results.get_data("Sequence")

        msr = abs(spec_res)
        # phase = np.unwrap(np.angle(spec_res))
        phase = np.angle(spec_res)
        i = spec_res.real
        q = spec_res.imag

        return msr, phase, i, q

    def execute_pulse_sequences(self, sequences):

        # if self.sequence == sequence:
        #     self.repeat_seq()
        # else:
        #     self.sequence_to_ZurichPulses(sequence)
        #     self.sequencePulses_to_exp()
        #     self.run_seq()

        self.sequences_to_ZurichPulses(sequences)
        self.sequencesPulses_to_exp()
        self.run_seq()

        spec_res = self.results.get_data("Sequence")

        msr = abs(spec_res)
        # phase = np.unwrap(np.angle(spec_res))
        phase = np.angle(spec_res)
        i = spec_res.real
        q = spec_res.imag

        return msr, phase, i, q

    def repeat_pulse_sequence(self, sequence):
        self.repeat_seq()

        spec_res = self.results.get_data("Sequence")

        msr = abs(spec_res)
        # phase = np.unwrap(np.angle(spec_res))
        phase = np.angle(spec_res)
        i = spec_res.real
        q = spec_res.imag

        return msr, phase, i, q

    # plot output signals [Zurich code] our pulse.plot() is better
    def plot_output_signals(self):
        """Plot output signals of a single-qubit experiment.

        Warning: Labelling the plots relies on assumptions valid for the example notebooks.
        """

        # Get a list of signals as a list, with one entry for each device involved
        signal_list = self.results.compiled_experiment.output_signals.signals

        # Count how many waveforms are in the experiment:
        n_rows = 0
        for signal in signal_list:
            n_rows += 1

        # Set up the plot
        fig, ax = plt.subplots(n_rows, 1, figsize=(5, 1.5 * n_rows), sharex=True)
        fig.subplots_adjust(hspace=0.4)

        rows = iter(range(n_rows))

        # Plot all signals of all involved devices. Some logic for correct labelling.
        for signal in signal_list:
            row = next(rows)

            device_uid = signal["device_uid"]
            n_channels = len(signal["channels"])

            for waveform in signal["channels"]:
                uid = waveform.uid
                time = waveform.time_axis

                if (
                    not "qa" in uid.lower() and not "_freq" in uid.lower()
                ):  # ignore QA triggers and oscillator frequency
                    title = ""
                    if "hdawg" in device_uid.lower() and n_channels == 1:
                        title = "Flux Pulse"
                    elif "qa" in device_uid.lower():
                        title = "Readout Pulse"
                    elif "qc" in device_uid.lower() and not "sg" in device_uid.lower():
                        title = "Readout Pulse"
                    else:
                        title = "Drive Pulse"

                    legend = None
                    if "sg" in device_uid.lower():
                        legend = "I" if "i" in uid.lower() else "Q"
                    elif "shfqa" in device_uid.lower() or "shfqc" in device_uid.lower():
                        pass
                    else:
                        try:
                            legend = "I" if int(uid) % 2 == 1 else "Q"
                        except:
                            pass

                    if n_rows > 1:
                        ax[row].plot(time, waveform.data, label=legend)
                        ax[row].set_title(title)
                        ax[row].set_ylabel("Amplitude")
                        ax[row].legend()
                    else:
                        ax.plot(time, waveform.data, label=legend)
                        ax.set_title(title)
                        ax.set_ylabel("Amplitude")
                        ax.legend()
                        ax.set_xlabel("Time (s)")
        if n_rows > 1:
            ax[-1].set_xlabel("Time (s)")

    # TODO: Add more gates
    def create_qubit_readout_pulse(self, qubit, start):
        ro_duration = self.native_gates["single_qubit"][qubit]["MZ"]["duration"]
        ro_frequency = self.native_gates["single_qubit"][qubit]["MZ"]["frequency"]
        ro_amplitude = self.native_gates["single_qubit"][qubit]["MZ"]["amplitude"]
        ro_shape = self.native_gates["single_qubit"][qubit]["MZ"]["shape"]
        ro_channel = self.qubit_channel_map[qubit][0]
        from qibolab.pulses import ReadoutPulse

        return ReadoutPulse(start, ro_duration, ro_amplitude, ro_frequency, 0, ro_shape, ro_channel, qubit=qubit)

    def create_qubit_drive_pulse(self, qubit, start, duration, relative_phase=0):
        qd_frequency = self.native_gates["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.native_gates["single_qubit"][qubit]["RX"]["amplitude"]
        qd_shape = self.native_gates["single_qubit"][qubit]["RX"]["shape"]
        qd_channel = self.qubit_channel_map[qubit][1]
        from qibolab.pulses import Pulse

        return Pulse(start, duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_RX_pulse(self, qubit, start=0, relative_phase=0):
        qd_duration = self.native_gates["single_qubit"][qubit]["RX"]["duration"]
        qd_frequency = self.native_gates["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.native_gates["single_qubit"][qubit]["RX"]["amplitude"]
        qd_shape = self.native_gates["single_qubit"][qubit]["RX"]["shape"]
        qd_channel = self.qubit_channel_map[qubit][1]
        from qibolab.pulses import Pulse

        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_RX90_pulse(self, qubit, start=0, relative_phase=0):
        qd_duration = int(self.native_gates["single_qubit"][qubit]["RX"]["duration"] / 2)
        qd_frequency = self.native_gates["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.native_gates["single_qubit"][qubit]["RX"]["amplitude"]
        qd_shape = self.native_gates["single_qubit"][qubit]["RX"]["shape"]
        qd_channel = self.qubit_channel_map[qubit][1]
        from qibolab.pulses import Pulse

        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_MZ_pulse(self, qubit, start):
        ro_duration = self.native_gates["single_qubit"][qubit]["MZ"]["duration"]
        ro_frequency = self.native_gates["single_qubit"][qubit]["MZ"]["frequency"]
        ro_amplitude = self.native_gates["single_qubit"][qubit]["MZ"]["amplitude"]
        ro_shape = self.native_gates["single_qubit"][qubit]["MZ"]["shape"]
        ro_channel = self.qubit_channel_map[qubit][0]
        from qibolab.pulses import ReadoutPulse

        return ReadoutPulse(start, ro_duration, ro_amplitude, ro_frequency, 0, ro_shape, ro_channel, qubit=qubit)

    def create_RX90_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        # create RX pi/2 pulse with drag shape
        qd_duration = self.native_gates["single_qubit"][qubit]["RX"]["duration"]
        qd_frequency = self.native_gates["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.native_gates["single_qubit"][qubit]["RX"]["amplitude"] / 2
        qd_shape = self.native_gates["single_qubit"][qubit]["RX"]["shape"]
        if beta != None:
            qd_shape = "Drag(5," + str(beta) + ")"

        qd_channel = self.qubit_channel_map[qubit][1]
        from qibolab.pulses import Pulse

        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)

    def create_RX_drag_pulse(self, qubit, start, relative_phase=0, beta=None):
        # create RX pi pulse with drag shape
        qd_duration = self.native_gates["single_qubit"][qubit]["RX"]["duration"]
        qd_frequency = self.native_gates["single_qubit"][qubit]["RX"]["frequency"]
        qd_amplitude = self.native_gates["single_qubit"][qubit]["RX"]["amplitude"]
        qd_shape = self.native_gates["single_qubit"][qubit]["RX"]["shape"]
        if beta != None:
            qd_shape = "Drag(5," + str(beta) + ")"

        qd_channel = self.qubit_channel_map[qubit][1]
        from qibolab.pulses import Pulse

        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, relative_phase, qd_shape, qd_channel, qubit=qubit)
