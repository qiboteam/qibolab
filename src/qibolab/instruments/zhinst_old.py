# -*- coding: utf-8 -*-
import laboneq.simple as lo
import matplotlib.pyplot as plt
import numpy as np
import yaml

from qibolab.instruments.abstract import AbstractInstrument, InstrumentException
from qibolab.pulses import PulseSequence



#TODO:ERASE
class SHFQC_QA(AbstractInstrument):
    
    """Instrument object for controlling Zurich Instruments (Zhinst) SHFQC controllers.
    Playing pulses on Zhinst controllers requires a ``Calibration`` object, a channel map and code
    written in Zhinst python library to define the experiment. 
    
    The signal ``Calibration`` for each qubit is generated in:
        -  def_calibration()
        from data retrived from the runcard in the setup() [``Qubit`` object in the future],
        and applied in the Zsetup() to the instrument.
        
    The Zhinst code for executing an arbitrary qibolab ``PulseSequence`` is written in
    several steps depending on how you want to run the pulse sequence or pulse sequences (Sweeps):
        1. Translation of qibo pulses to zhinst pulses (sequence_to_ZurichPulses, sequences_to_ZurichPulses_np, sequences_to_ZurichPulses)
        2. Assign the zhinst pulses to a zhinst experiment (sequencePulses_to_exp, sequencesPulses_to_exp, sequencesPulses_to_exp_np, sequencePulses_to_exp_freqs, sequencePulses_to_exp_Sweep)
        3. Execute the Pulse sequence and retrieve the results after the experiment ends in a qibocal fashion (execute_pulse_sequence_NoSamples, execute_pulse_sequences, execute_pulse_sequences_np, execute_pulse_sequence_freq, execute_pulse_sequence_Sweep)
    
    Translation of pulses for now is limited by the currently defined functions to Readout[Rectangular], Drive [Rectangular, Gaussian, Drag] but it can be expanded.
    1. (Option b, Avoid) Retrieve pulse samples from qibo pulse and rebuild the pulse sequence as a zhinst pulse sequence 
    
    
    Options:
        Single Pulse:
            - execute_pulse_sequence_NoSamples:  Regular arbitrary single pulse sequence execution
        Sweep from qibo pulse sequences:
            - execute_pulse_sequence_freq: Frequency sweeps need a unique zhinst experiment definition
            - execute_pulse_sequences: Regular sweep of the pulse parameters (Lenght, Amplitude, Phase, Delays, etc)
            - execute_pulse_sequences_np: [TODO] Same as above but with np.array or dictionaries as for big sequences the lists get slow
            - execute_pulse_sequence_Sweep: [WIP] Same as above but in a more zhinst way to get rid of lists
    
        Sweep from qibo Sweep:
            - execute_Sweep: [WIP]
    
        TODO: Set the acquisition_type optimally.
    
    Args:
        name (str): Name of the instrument instance.
        address (str): Zhinst address for connecting to the instruments inside their zhinst DataServer.
        runcard (str): Runcard location with qubit parameters [Chnage to qubit object ?]
        use_emulation (bool): Emulate or not
        
    Attributes:
        is_connected (bool): Boolean that shows whether instruments are connected.
        TODO: add all the rest
        
    """
    
    def __init__(self, name, address, runcard, use_emulation):
        super().__init__(name, address)
        self.device = None

        self.channels = None
        self.lo_frequency = None  # in units of Hz
        self.gain = None
        self.input_range = None  # in units of dBm
        self.output_range = None  # in units of dBm

        self._latest_sequence = None
        # Hardcoded file that describes the experimental setup and channels
        self.descriptor_path = "/home/admin/Juan/qibolab/src/qibolab/runcards/descriptor_shfqc.yml"
        self.runcard_file = runcard
        self.emulation = use_emulation

        with open(runcard) as file:
            settings = yaml.safe_load(file)

        self.setup(**settings)
        self.def_calibration()
        self.Z_setup()
        self.connect(use_emulation=self.emulation)


    # For first tries and debugging, exp(zhinst experiment object): Experiment
    def compile_exp(self, exp):
        self.exp = self.session.compile(exp)

    def run_exp(self):
        self.results = self.session.run(self.exp)

    def run_seq(self):
        self.exp = self.session.compile(self.experiment)
        self.results = self.session.run(self.exp, self.emulation)
        # self.results = self.session.run(self.experiment, self.emulation)

    def repeat_seq(self):
        self.results = self.session.run(do_simulation=self.emulation)
        # self.results = self.session.run(self.experiment, self.emulation)

    # TODO: Proper calibration and channel map
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
    def set_map(self):
        
        self.map_q = {}
        
        for qubit in [0]:
            if any(self.sequence_drive):
                self.map_q[f"drive{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["drive_line"]
                self.map_q[f"measure{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["measure_line"]
                self.map_q[f"acquire{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["acquire_line"]
                
            else:
                self.map_q[f"measure{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["measure_line"]
                self.map_q[f"acquire{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["acquire_line"]  
                

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

    # Reload settings 
    def reload_settings(self):
        with open(self.runcard_file) as file:
            self.settings = yaml.safe_load(file)
        if self.is_connected:
            self.setup(**self.settings)
            
    # Apply reloaded settings (join ?)   
    def apply_settings(self):
        self.def_calibration()
        self.Zsetup.set_calibration(self.calib)

    # FIXME: What are these for ???
    def start(self):
        pass

    def stop(self):
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

    # TODO: Can we sweep multiple parameters ?
    def sequence_to_ZurichSweep(self, sequence, start, stop, count, parameter):

        self.sequence = sequence
        sweep_parameters = []
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

        sweep_parameter = lo.LinearSweepParameter(uid="Lenght", start=start * 1e-9, stop=stop * 1e-9, count=count)

        self.SweepParameters = sweep_parameter
        self.Parameter = parameter
        # self.SweepParameters = sweep_parameters
        self.delays = delays
        self.sequence_drive = sequence_Z_drive
        self.sequence_readout = sequence_Z_readout

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

    # Can't get the arrays ordered and divided as I want so I made list and turn them into a np.array
    # Runs as expected but no improvements
    def sequences_to_ZurichPulses_np(self, sequences):

        self.sequences = sequences

        sequence_Z_drives = []
        sequence_Z_readouts = []
        sequence_Z_weights = []
        Delays = []
        rel_phases = []
        Drive_durations = []

        for k in range(len(sequences)):
            sequence = sequences[k]

            sequence_Z_drive = []
            sequence_Z_readout = []
            sequence_Z_weight = []
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
                                uid=("drive" + str(k) + "_" + str(i)),
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
                                uid=("drive" + str(k) + "_" + str(i)),
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

                        sequence_Z_weight.append(
                            lo.pulse_library.const(
                                uid="readout_weighting_function" + str(k) + "_" + str(j),
                                length=2 * pulse.duration * 1e-9,
                                amplitude=1.0,
                            )
                        )

                j += 1

                delays = []
                for i in range(len(starts) - 1):
                    delays.append(starts[i + 1] - durations[i])

            Drive_durations.append(Drive_duration)
            sequence_Z_readouts.append(sequence_Z_readout)
            sequence_Z_weights.append(sequence_Z_weight)
            sequence_Z_drives.append(sequence_Z_drive)
            Delays.append(delays)
            rel_phases.append(rel_phase)

        self.delays = np.asarray(Delays)
        self.sequence_drive = np.asarray(sequence_Z_drives)
        self.sequence_readout = np.asarray(sequence_Z_readouts)
        self.sequence_weight = np.asarray(sequence_Z_weights)
        self.rel_phases = np.asarray(rel_phases)
        self.Drive_durations = np.asarray(Drive_durations)

    def sequences_to_ZurichPulses(self, sequences):

        self.sequences = sequences

        sequence_Z_drives = []
        sequence_Z_readouts = []
        sequence_Z_weights = []
        Delays = []
        rel_phases = []
        Drive_durations = []

        for k in range(len(sequences)):
            sequence = sequences[k]

            sequence_Z_drive = []
            sequence_Z_readout = []
            sequence_Z_weight = []
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
                                uid=("drive" + str(k) + "_" + str(i)),
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
                                uid=("drive" + str(k) + "_" + str(i)),
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

                        sequence_Z_weight.append(
                            lo.pulse_library.const(
                                uid="readout_weighting_function" + str(k) + "_" + str(j),
                                length=2 * pulse.duration * 1e-9,
                                amplitude=1.0,
                            )
                        )

                j += 1

                delays = []
                for i in range(len(starts) - 1):
                    delays.append(starts[i + 1] - durations[i])

            Drive_durations.append(Drive_duration)
            sequence_Z_readouts.append(sequence_Z_readout)
            sequence_Z_weights.append(sequence_Z_weight)
            sequence_Z_drives.append(sequence_Z_drive)
            Delays.append(delays)
            rel_phases.append(rel_phase)

        self.delays = Delays
        self.sequence_drive = sequence_Z_drives
        self.sequence_readout = sequence_Z_readouts
        self.sequence_weight = sequence_Z_weights
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

            ## experimental pulse sequence
            # outer loop - real-time, cyclic averaging in standard integration mode
            with exp.acquire_loop_rt(
                count=self.settings["hardware_avg"],
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                averaging_mode=lo.AveragingMode.CYCLIC,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):

                for j in range(len(self.sequence_readout)):
                    sequence_d = self.sequence_drive[j]
                    sequence_r = self.sequence_readout[j]
                    sequence_w = self.sequence_weight[j]

                    # # inner loop - real-time sweep of qubit drive pulse amplitude
                    with exp.section(uid=f"sequence{j}_drive", alignment=lo.SectionAlignment.RIGHT):
                        i = 0
                        for pulse in sequence_d:
                            exp.play(signal="drive", pulse=pulse, phase=self.rel_phases[j][i])
                            if self.delays[j][i] > 0:
                                exp.delay(signal="drive", time=self.delays[j][i] * 1e-9)
                            i += 1

                    # qubit readout pulse and data acquisition
                    with exp.section(uid=f"sequence{j}_measure"):
                        exp.reserve(signal="drive")
                        exp.play(signal="measure", pulse=sequence_r[0], phase=self.rel_phases[j][i])
                        exp.acquire(signal="acquire", handle=f"sequence{j}", kernel=sequence_w[0])

                        # integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]
                        # exp.acquire(signal="acquire", handle=f"sequence{j}", length=integration_time)
                        
                    # relax time after readout - for signal processing and qubit relaxation to ground state
                    with exp.section(uid=f"sequence{j}_relax", length=self.settings["readout_delay"]):
                        exp.reserve(signal="drive")
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

            self.map_q = {
                "measure": self.Zsetup.logical_signal_groups["q"].logical_signals["measure_line"],
                "acquire": self.Zsetup.logical_signal_groups["q"].logical_signals["acquire_line"],
            }

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

                # readout_weighting_function = lo.pulse_library.const(
                #     uid="readout_weighting_function",
                #     length=self.sequence_readout[0].duration * 1e-9,
                #     amplitude=1.0,
                # )

                for j in range(len(self.sequence_readout)):
                    sequence_r = self.sequence_readout[j]

                    with exp.section(
                        length=5e-6, alignment=lo.SectionAlignment.RIGHT, uid=f"sequence{j}_qubit_readout"
                    ):
                        # for pulse in self.sequence_readout:
                        # exp.play(signal="measure", pulse=pulse)
                        i = 0
                        exp.play(signal="measure", pulse=sequence_r[i], phase=self.rel_phases[j][i])

                        integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]

                        exp.acquire(signal="acquire", handle=f"sequence{j}", length=integration_time)

                        # exp.acquire(
                        #     signal="acquire",
                        #     handle=f"sequence{j}",
                        #     kernel=self.readout_weighting_function,
                        # )

                    # relax time after readout - for signal processing and qubit relaxation to ground state
                    with exp.section(uid=f"sequence{j}_relax"):
                        exp.delay(signal="measure", time=self.settings["readout_delay"])

        self.set_map()
        exp.set_signal_map(self.map_q)
        self.experiment = exp

    def sequencesPulses_to_exp_np(self):
        # Create Experiment
        # if len(self.sequence_drive[0]) != 0:
        if self.sequence_drive.shape[0] != 0:
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
                count=self.settings["hardware_avg"],
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                averaging_mode=lo.AveragingMode.CYCLIC,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):

                # for j in range(len(self.sequence_readout)):
                for j in range(self.sequence_readout.shape[0]):
                    sequence_d = self.sequence_drive[j]
                    sequence_r = self.sequence_readout[j]
                    sequence_w = self.sequence_weight[j]

                    # # inner loop - real-time sweep of qubit drive pulse amplitude
                    with exp.section(uid=f"sequence{j}_drive", alignment=lo.SectionAlignment.RIGHT):
                        i = 0
                        for pulse in sequence_d:
                            exp.play(signal="drive", pulse=pulse, phase=self.rel_phases[j][i])
                            if self.delays[j][i] > 0:
                                exp.delay(signal="drive", time=self.delays[j][i] * 1e-9)
                            i += 1

                    # qubit readout pulse and data acquisition
                    with exp.section(uid=f"sequence{j}_measure"):
                        exp.reserve(signal="drive")
                        exp.play(signal="measure", pulse=sequence_r[0], phase=self.rel_phases[j][i])

                        # integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]
                        # exp.acquire(signal="acquire", handle=f"sequence{j}", length=integration_time)

                        exp.acquire(signal="acquire", handle=f"sequence{j}", kernel=sequence_w[0])

                    # relax time after readout - for signal processing and qubit relaxation to ground state
                    with exp.section(uid=f"sequence{j}_relax", length=self.settings["readout_delay"]):
                        exp.reserve(signal="drive")
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

            self.map_q = {
                "measure": self.Zsetup.logical_signal_groups["q"].logical_signals["measure_line"],
                "acquire": self.Zsetup.logical_signal_groups["q"].logical_signals["acquire_line"],
            }

            ## experimental pulse sequence
            # outer loop - real-time, cyclic averaging in standard integration mode
            with exp.acquire_loop_rt(
                uid="shots",
                count=self.settings["hardware_avg"],
                averaging_mode=lo.AveragingMode.CYCLIC,
                # averaging_mode=lo.AveragingMode.SEQUENTIAL,
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

                    with exp.section(
                        length=5e-6, alignment=lo.SectionAlignment.RIGHT, uid=f"sequence{j}_qubit_readout"
                    ):
                        # for pulse in self.sequence_readout:
                        # exp.play(signal="measure", pulse=pulse)
                        i = 0
                        exp.play(signal="measure", pulse=sequence_r[i], phase=self.rel_phases[j][i])

                        integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]

                        exp.acquire(signal="acquire", handle=f"sequence{j}", length=integration_time)

                        # exp.acquire(
                        #     signal="acquire",
                        #     handle=f"sequence{j}",
                        #     kernel=self.readout_weighting_function,
                        # )

                    # relax time after readout - for signal processing and qubit relaxation to ground state
                    with exp.section(uid=f"sequence{j}_relax"):
                        exp.delay(signal="measure", time=self.settings["readout_delay"])

        self.set_map()
        exp.set_signal_map(self.map_q)
        self.experiment = exp

    def sequencePulses_to_exp_freqs(self, start, stop, points):
        # Create Experiment

        frequency_sweep = lo.LinearSweepParameter(uid=f"frequency_sweep", start=start, stop=stop, count=points)

        if len(self.sequence_drive) != 0:
            exp = lo.Experiment(
                uid="Sequence",
                signals=[
                    lo.ExperimentSignal("drive"),
                    lo.ExperimentSignal("measure"),
                    lo.ExperimentSignal("acquire"),
                ],
            )

            self.map_q = {
                "drive": self.Zsetup.logical_signal_groups["q0"].logical_signals["drive_line"],
                "measure": self.Zsetup.logical_signal_groups["q"].logical_signals["measure_line"],
                "acquire": self.Zsetup.logical_signal_groups["q"].logical_signals["acquire_line"],
            }

            ## experimental pulse sequence
            # outer loop - real-time, cyclic averaging in standard integration mode
            with exp.acquire_loop_rt(
                count=self.settings["hardware_avg"],
                averaging_mode=lo.AveragingMode.SEQUENTIAL,
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                # averaging_mode=lo.AveragingMode.CYCLIC,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):

                with exp.sweep(parameter=frequency_sweep):
                    j = 0
                    # # inner loop - real-time sweep of qubit drive pulse amplitude
                    # with exp.sweep(uid="sweep", parameter=sweep_rel_flat, alignment=SectionAlignment.RIGHT):
                    # qubit excitation - pulse amplitude will be swept
                    with exp.section(alignment=lo.SectionAlignment.RIGHT):
                        i = 0
                        for pulse in self.sequence_drive:
                            exp.play(signal="drive", pulse=pulse, phase=self.rel_phases[i])

                            if self.delays[i] > 0:
                                exp.delay(signal="drive", time=self.delays[i] * 1e-9)
                            i += 1

                        # exp.delay(signal="drive", time=300e-9)

                    # qubit readout pulse and data acquisition
                    with exp.section():
                        for pulse in self.sequence_readout:

                            exp.reserve(signal="drive")
                            exp.play(signal="measure", pulse=pulse, phase=self.rel_phases[i])

                            integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]

                            exp.acquire(signal="acquire", handle=f"sequence{j}", length=integration_time)

                            j += 1

                    # relax time after readout - for signal processing and qubit relaxation to ground state
                    with exp.section():
                        exp.reserve(signal="drive")
                        exp.delay(signal="measure", time=self.settings["readout_delay"])

            # define experiment calibration - sweep over qubit drive frequency
            exp_calib = lo.Calibration()
            exp_calib["drive"] = lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=frequency_sweep,
                    modulation_type=lo.ModulationType.HARDWARE,
                )
            )

        # TODO: Add features of above to else
        else:
            exp = lo.Experiment(
                uid="Sequence",
                signals=[
                    lo.ExperimentSignal("measure"),
                    lo.ExperimentSignal("acquire"),
                ],
            )

            self.map_q = {
                "measure": self.Zsetup.logical_signal_groups["q"].logical_signals["measure_line"],
                "acquire": self.Zsetup.logical_signal_groups["q"].logical_signals["acquire_line"],
            }

            ## experimental pulse sequence
            # outer loop - real-time, cyclic averaging in standard integration mode
            with exp.acquire_loop_rt(
                count=self.settings["hardware_avg"],
                # averaging_mode=lo.AveragingMode.CYCLIC,
                averaging_mode=lo.AveragingMode.SEQUENTIAL,
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):
                # # inner loop - real-time sweep of qubit drive pulse amplitude
                # qubit readout pulse and data acquisition
                with exp.sweep(parameter=frequency_sweep):
                    j = 0
                    with exp.section(alignment=lo.SectionAlignment.RIGHT):
                        i = 0
                        for pulse in self.sequence_readout:
                            exp.play(signal="measure", pulse=pulse, phase=self.rel_phases[i])
                            integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]
                            exp.acquire(
                                signal="acquire",
                                handle=f"sequence{j}",
                                length=integration_time,
                            )
                            j += 1

                    # relax time after readout - for signal processing and qubit relaxation to ground state
                    with exp.section():
                        exp.delay(signal="measure", time=self.settings["readout_delay"])

            # define experiment calibration - sweep over qubit drive frequency
            exp_calib = lo.Calibration()
            exp_calib["measure"] = lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=frequency_sweep,
                    modulation_type=lo.ModulationType.HARDWARE,
                )
            )

        exp.set_calibration(exp_calib)

        # self.set_map()
        exp.set_signal_map(self.map_q)
        self.experiment = exp

    def sequencePulses_to_exp_Sweep(self):
        # Create Experiment

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

            with exp.sweep(uid="sweep", parameter=self.SweepParameters, alignment=lo.SectionAlignment.RIGHT):

                with exp.section(uid="qubit_excitation", alignment=lo.SectionAlignment.RIGHT):
                    i = 0
                    for pulse in self.sequence_drive:

                        if self.Parameter == "Lenght":

                            exp.play(signal="drive", pulse=pulse, length=self.SweepParameters, phase=self.rel_phases[i])

                        if self.Parameter == "Amp":

                            exp.play(
                                signal="drive", pulse=pulse, amplitude=self.SweepParameters, phase=self.rel_phases[i]
                            )

                        # if self.delays[i] > 0:
                        #     exp.delay(signal="drive", time=self.delays[i] * 1e-9)
                        # i += 1

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

        map_q = {
            "drive": self.Zsetup.logical_signal_groups["q0"].logical_signals["drive_line"],
            "measure": self.Zsetup.logical_signal_groups["q"].logical_signals["measure_line"],
            "acquire": self.Zsetup.logical_signal_groups["q"].logical_signals["acquire_line"],
        }

        # self.set_map()
        exp.set_signal_map(map_q)
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
                i = 0
                with exp.section(uid="qubit_readout"):
                    for pulse in self.sequence_readout:
                        exp.play(signal="measure0", pulse=pulse, phase=self.rel_phases[i])

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

        self.set_map()
        exp.set_signal_map(self.map_q)

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

        spec_res = self.results.get_data("sequence")

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

        spec_res = []
        msr = []
        phase = []
        i = []
        q = []
        for j in range(len(self.sequence_readout)):
            spec_res.append(self.results.get_data(f"sequence{j}"))
            msr.append(abs(spec_res[j]))
            phase.append(np.angle(spec_res[j]))
            i.append(spec_res[j].real)
            q.append(spec_res[j].imag)

        return msr, phase, i, q

    def execute_pulse_sequences_np(self, sequences):

        # if self.sequence == sequence:
        #     self.repeat_seq()
        # else:
        #     self.sequence_to_ZurichPulses(sequence)
        #     self.sequencePulses_to_exp()
        #     self.run_seq()

        self.sequences_to_ZurichPulses_np(sequences)
        self.sequencesPulses_to_exp_np()
        self.run_seq()

        spec_res = []
        msr = []
        phase = []
        i = []
        q = []
        for j in range(len(self.sequence_readout)):
            spec_res.append(self.results.get_data(f"sequence{j}"))
            msr.append(abs(spec_res[j]))
            phase.append(np.angle(spec_res[j]))
            i.append(spec_res[j].real)
            q.append(spec_res[j].imag)

        return msr, phase, i, q

    def execute_pulse_sequence_freq(self, sequence, start, stop, points):

        # if self.sequence == sequence:
        #     self.repeat_seq()
        # else:
        #     self.sequence_to_ZurichPulses(sequence)
        #     self.sequencePulses_to_exp()
        #     self.run_seq()

        self.sequence_to_ZurichPulses(sequence)
        self.sequencePulses_to_exp_freqs(start, stop, points)
        self.run_seq()

        spec_res = []
        msr = []
        phase = []
        i = []
        q = []
        for j in range(len(self.sequence_readout)):
            spec_res.append(self.results.get_data(f"sequence{j}"))
            msr.append(abs(spec_res[j]))
            phase.append(np.angle(spec_res[j]))
            i.append(spec_res[j].real)
            q.append(spec_res[j].imag)

        return msr, phase, i, q

    def execute_pulse_sequence_Sweep(self, sequence, start, stop, points, parameter):

        # if self.sequence == sequence:
        #     self.repeat_seq()
        # else:
        #     self.sequence_to_ZurichPulses(sequence)
        #     self.sequencePulses_to_exp()
        #     self.run_seq()

        self.sequence_to_ZurichSweep(sequence, start, stop, points, parameter)
        self.sequencePulses_to_exp_Sweep()
        self.run_seq()

        spec_res = []
        msr = []
        phase = []
        i = []
        q = []
        for j in range(points):
            spec_res.append(self.results.get_data(f"sequence")[j])
            msr.append(abs(spec_res[j]))
            phase.append(np.angle(spec_res[j]))
            i.append(spec_res[j].real)
            q.append(spec_res[j].imag)

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
        
#TODO: Add/Check for loops for multiple qubits
class Zurich(AbstractInstrument):
    
    def __init__(self, name, address, runcard, use_emulation):
        super().__init__(name, address)

        # Hardcoded file that describes the experimental setup and channels
        self.descriptor_path = "/home/admin/Juan/qibolab/src/qibolab/runcards/descriptor_shfqc_hdawg.yml"
        self.runcard_file = runcard
        self.emulation = use_emulation

        with open(runcard) as file:
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
    
    def repeat_seq(self):
        self.results = self.session.run(do_simulation=self.emulation)
        
    def run_multi(self):
        
        compiler_settings={
            "SHFSG_FORCE_COMMAND_TABLE": True,
            "SHFSG_MIN_PLAYWAVE_HINT": 32,
            "SHFSG_MIN_PLAYZERO_HINT": 32,
        }
        
        self.exp = self.session.compile(self.experiment,  compiler_settings=compiler_settings)
        self.results = self.session.run(self.exp, self.emulation)
    
    def def_calibration(self):
        
        self.calib = lo.Calibration()
        
        for it in range(len(self.qubits)):
            qubit = self.qubits[it]
        
            self.calib[f"/logical_signal_groups/q{qubit}/measure_line"] = lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=self.instruments["shfqc_qa"]["settings"][f"if_frequency_{qubit}"],
                    modulation_type=lo.ModulationType.HARDWARE,
                    # modulation_type=lo.ModulationType.SOFTWARE,
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
                
            self.calib[f"/logical_signal_groups/q{qubit}/flux_line"] = lo.SignalCalibration(
                # modulation_type=lo.ModulationType.HARDWARE,
                range=self.instruments["hdawg"]["settings"]["flux_range"],
                port_delay=0,  # applied to corresponding instrument node, bound to hardware limits
                delay_signal=0,
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
                    self.map_q[f"drive{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["drive_line"]
                    self.map_q[f"flux{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["flux_line"]
                    self.map_q[f"measure{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["measure_line"]
                    self.map_q[f"acquire{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["acquire_line"]
                else:
                    self.map_q[f"drive{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["drive_line"]
                    self.map_q[f"measure{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["measure_line"]
                    self.map_q[f"acquire{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["acquire_line"]
                
            elif any(self.sequence_flux):
                self.map_q[f"flux{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["flux_line"]
                self.map_q[f"measure{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["measure_line"]
                self.map_q[f"acquire{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["acquire_line"]
                
            else:
                self.map_q[f"measure{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["measure_line"]
                self.map_q[f"acquire{qubit}"] = self.Zsetup.logical_signal_groups[f"q{qubit}"].logical_signals["acquire_line"]  
                
    def connect(self, use_emulation):
        if not self.is_connected:
            for attempt in range(3):
                try:
                    self.session = lo.Session(self.Zsetup)
                    # self.device = self.session.connect(self.address)
                    self.device = self.session.connect(do_emulation=use_emulation)
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
            self.session = Session("localhost")
            self.device = self.session.disconnect_device(self.address)
            self.is_connected = False
            global cluster
            cluster = None
        else:
            print(f"Already disconnected")
          
    #Join Setups
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

    # Reload settings 
    def reload_settings(self):
        with open(self.runcard_file) as file:
            self.settings = yaml.safe_load(file)
        if self.is_connected:
            self.setup(**self.settings)
    def apply_settings(self):
        self.def_calibration()
        self.Zsetup.set_calibration(self.calib) 

    def sequences_to_ZurichPulses(self, sequences, sweepers = None):
        
        self.sequences = sequences
        
        sequence_Z_drives = []
        sequence_Z_readouts = []
        sequence_Z_weights = []
        sequence_Z_fluxs = []
        Delays = []
        rel_phases = []
        Drive_durations = []
        addressed_qubits = []
        
        # if len(sequences) == 1:
        #     addressed_qubits = []
        # else: 
        #     sequence_Z_drives = []
        #     sequence_Z_readouts = []
        #     sequence_Z_weights = []
        #     sequence_Z_fluxs = []
        #     Delays = []
        #     rel_phases = []
        #     Drive_durations = []
        #     addressed_qubits = []

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
                                uid=(f"drive_{qubit}_" + str(l) + "_"  + str(i)),
                                length=pulse.duration * 1e-9,
                                amplitude=pulse.amplitude,
                            )
                        )
                    elif "Gaussian" in str(pulse.shape):
                        sigma = str(pulse.shape).removeprefix("Gaussian(")
                        sigma = float(sigma.removesuffix(")"))
                        sequence_Z_drive.append(
                            lo.pulse_library.gaussian(
                                uid=(f"drive_{qubit}_" + str(l) + "_" + str(i)),
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
                                uid=(f"drive_{qubit}_" + str(l) + "_" + str(i)),
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
                                uid=(f"readout_{qubit}_" + str(l) + "_" + str(j)),
                                length=pulse.duration * 1e-9,
                                amplitude=pulse.amplitude,
                            )
                        )

                        sequence_Z_weight.append(
                            lo.pulse_library.const(
                                uid="readout_weighting_function" + str(l) + "_" + str(j),
                                length=2 * pulse.duration * 1e-9,
                                amplitude=1.0,
                            )
                        )
                    j += 1
                
                if str(pulse.type) == "PulseType.FLUX":
                    if str(pulse.shape) == "Rectangular()":
                        sequence_Z_flux.append(
                            lo.pulse_library.const(
                                uid=(f"flux_{qubit}_"+ str(l) + "_" + str(k)),
                                length=pulse.duration * 1e-9,
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
                

        self.delays = Delays
        self.sequence_drive = sequence_Z_drives
        self.sequence_readout = sequence_Z_readouts
        self.sequence_flux = sequence_Z_fluxs
        self.sequence_weight = sequence_Z_weights
        self.rel_phases = rel_phases
        self.Drive_durations = Drive_durations
        self.addressed_qubits = addressed_qubits
            
            # if len(sequences) == 1:
            #     self.delays = delays
            #     self.sequence_drive = sequence_Z_drive
            #     self.sequence_readout = sequence_Z_readout
            #     self.sequence_flux = sequence_Z_flux
            #     self.sequence_weight = sequence_Z_weight
            #     self.rel_phases = rel_phase
            #     self.Drive_durations = Drive_duration
            #     self.addressed_qubits = addressed_qubits
            # else: 
            #     Drive_durations.append(Drive_duration)
            #     sequence_Z_fluxs.append(sequence_Z_flux)
            #     sequence_Z_readouts.append(sequence_Z_readout)
            #     sequence_Z_weights.append(sequence_Z_weight)
            #     sequence_Z_drives.append(sequence_Z_drive)
            #     Delays.append(delays)
            #     rel_phases.append(rel_phase)
                
        # if len(sequences)d_qubits = addressed_qubits
            
        self.sweepers = sweepers
        if sweepers != None:
            self.sweepers = sweepers
            sweepers_Zh = []
            for sweep in sweepers:
                sweepers_Zh.append(lo.LinearSweepParameter(uid=sweep.parameter, start=sweep.start, stop=sweep.stop, count=sweep.count)) 
            self.sweepers_Zh = sweepers_Zh

    def create_exp(self):

        signals = []
        if any(self.sequence_drive):
                if any(self.sequence_flux):
                    for qubit in self.addressed_qubits:
                        signals.append(lo.ExperimentSignal(f"drive{qubit}"))
                        signals.append(lo.ExperimentSignal(f"flux{qubit}"))
                        signals.append(lo.ExperimentSignal(f"measure{qubit}"))
                        signals.append(lo.ExperimentSignal(f"acquire{qubit}"))
        elif any(self.sequence_flux):
            for qubit in self.addressed_qubits:
                signals.append(lo.ExperimentSignal(f"drive{qubit}"))
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
        
        with exp.acquire_loop_rt(
            uid="shots",
            count=self.settings["hardware_avg"],
            acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
            averaging_mode=lo.AveragingMode.CYCLIC,
            # acquisition_type=lo.AcquisitionType.INTEGRATION,
        ):
        
            if any(self.sequence_drive):
                if any(self.sequence_flux):
                    for j in range(len(self.sequences)):
                        self.iteration = j
                        self.Flux(exp)
                        self.Drive(exp)
                        self.Measure(exp)
                        self.qubit_reset(exp)

            elif any(self.sequence_flux):
                
                for j in range(len(self.sequences)):
                    self.iteration = j
                    self.Drive(exp)
                    self.Measure(exp)
                    self.qubit_reset(exp)

            else:
        
                for j in range(len(self.sequences)):
                    with exp.section(uid=f"sequence{j}_measure"):
                        i = 0
                        for pulse in self.sequence_readout[j]:
                            qubit = pulse.uid.split("_")[1]
                            exp.play(signal=f"measure{qubit}", pulse=pulse, phase=self.rel_phases[j][i])
                            
                            integration_time = self.native_gates["single_qubit"][int(qubit)]["MZ"]["integration_time"]
                            exp.acquire(signal=f"acquire{qubit}", handle=f"sequence{j}", length=integration_time)
                            # exp.acquire(signal=f"acquire{qubit}", handle=f"sequence{j}_{k}", kernel=self.sequence_weight[j][i])
                            i += 1
                            
                    with exp.section(uid=f"relax", length=self.settings["readout_delay"]):
                        for qubit in self.addressed_qubits:
                            exp.delay(signal=f"measure{qubit}", time=self.settings["readout_delay"])
        
            
        #     if self.sweepers != None:
        #         if self.sweepers[0].parameter == "freq":     
        #             with exp.sweep(parameter=self.sweepers_Zh[0]):
        #                 k = 0
                        
                    
        #                 if any(self.sequence_drive):
        #                     if any(self.sequence_flux):
        #                         for j in range(len(self.sequences)):
        #                             self.iteration = j
        #                             self.Flux(exp)
        #                             self.Drive(exp)
        #                             self.Measure(exp)
        #                             self.qubit_reset(exp)

        #                 elif any(self.sequence_flux):
                            
        #                     for j in range(len(self.sequences)):
        #                         self.iteration = j
        #                         self.Drive(exp)
        #                         self.Measure(exp)
        #                         self.qubit_reset(exp)

        #                 else:
                            
        #                     for j in range(len(self.sequences)):
        #                         with exp.section(uid=f"sequence{j}_measure"):
        #                             i = 0
        #                             for pulse in self.sequence_readout[j]:
        #                                 qubit = pulse.uid.split("_")[1]
        #                                 exp.play(signal=f"measure{qubit}", pulse=pulse, phase=self.rel_phases[j][i])
                                        
        #                                 integration_time = self.native_gates["single_qubit"][qubit]["MZ"]["integration_time"]
        #                                 exp.acquire(signal=f"acquire{qubit}", handle=f"sequence{j}_{k}", length=integration_time)
        #                                 # exp.acquire(signal=f"acquire{qubit}", handle=f"sequence{j}_{k}", kernel=self.sequence_weight[j][i])
        #                                 i += 1
        #                             k += 1
                                     
        #                         with exp.section(uid=f"relax", length=self.settings["readout_delay"]):
        #                             for qubit in self.addressed_qubits:
        #                                 exp.delay(signal=f"measure{qubit}", time=self.settings["readout_delay"])
                               
                                
        #                         # self.iteration = j
        #                         # exp = self.Measure(exp)
        #                         # exp = self.qubit_reset(exp)
        #                         # self.Measure(exp)
        #                         # self.qubit_reset(exp)
                           
            
        #     else:
        #         if any(self.sequence_drive):
        #             if any(self.sequence_flux):
        #                 for j in range(len(self.sequences)):
        #                     self.iteration = j
        #                     self.Flux(exp)
        #                     self.Drive(exp)
        #                     self.Measure(exp)
        #                     self.qubit_reset(exp)

        #         elif any(self.sequence_flux):
                    
        #             for j in range(len(self.sequences)):
        #                 self.iteration = j
        #                 self.Drive(exp)
        #                 self.Measure(exp)
        #                 self.qubit_reset(exp)

        #         else:
                    
        #             for j in range(len(self.sequences)):
        #                 self.iteration = j
        #                 self.Measure(exp)
        #                 self.qubit_reset(exp)
                    

            
        #     if self.sweepers != None:
        #         if self.sweepers[0].parameter == "freq":  
        #             # define experiment calibration - sweep over qubit drive frequency
        #             exp_calib = lo.Calibration()
        #             exp_calib[f"measure{qubit}"] = lo.SignalCalibration(
        #                 oscillator=lo.Oscillator(
        #                     frequency=self.sweepers_Zh[0],
        #                     modulation_type=lo.ModulationType.HARDWARE,
        #                 )
        #             )
        
        self.set_maps()
        exp.set_signal_map(self.map_q)
        
        self.experiment = exp

    def Flux(self,exp):
        j = self.iteration
        with exp.section(uid=f"sequence{j}_flux_bias", alignment=lo.SectionAlignment.RIGHT):
            for pulse in self.sequence_flux[j]:
                qubit = pulse.uid.split("_")[1]
                exp.play(signal=f"flux{qubit}", pulse=pulse)

    def Drive(self, exp):
        j = self.iteration
        with exp.section(uid=f"sequence{j}_drive", alignment=lo.SectionAlignment.RIGHT):
            i = 0
            # for qubit in self.addressed_qubits:
            # exp.delay(signal=f"drive{qubit}", time = 10e-9) #ramp up
            for pulse in self.sequence_drive[j]:
                qubit = pulse.uid.split("_")[1]
                exp.play(signal=f"drive{qubit}", pulse=pulse, phase=self.rel_phases[j][i])

                if self.delays[j][i] > 0:
                    qubit = pulse.uid.split("_")[1]
                    exp.delay(signal=f"drive{qubit}", time=self.delays[j][i] * 1e-9)
                i += 1

    def Measure(self,exp):
         # qubit readout pulse and data acquisition
        j = self.iteration
        with exp.section(uid=f"sequence{j}_measure"):
            i = 0
            for pulse in self.sequence_readout[j]:
                qubit = pulse.uid.split("_")[1]
                exp.play(signal=f"measure{qubit}", pulse=pulse, phase=self.rel_phases[j][i])
                exp.acquire(signal=f"acquire{qubit}", handle=f"sequence{j}", kernel=self.sequence_weight[j][i])
                i += 1

        return exp

    def qubit_reset(self, exp):
        # relax time or fast reset after readout - for signal processing and qubit relaxation to ground state
        if self.settings["Fast_reset"] == True:
            for qubit in self.addressed_qubits:
                with exp.section(uid=f"fast_reset", length = self.settings["Fast_reset_time"]):
                    with exp.match_local(handle=f"acquire{qubit}"):
                        with exp.case(state=0):
                            pass
                            # exp.play(some_pulse)
                        with exp.case(state=1):
                            pass
                            # exp.play(some_other_pulse)
        else:
            with exp.section(uid=f"relax", length=self.settings["readout_delay"]):
                for qubit in self.addressed_qubits:
                    # exp.reserve(signal=f"drive{qubit}")
                    # exp.reserve(signal=f"flux{qubit}")
                    exp.delay(signal=f"measure{qubit}", time=self.settings["readout_delay"])
       
        return exp 

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

    def sequence_to_ZurichSweep_freq_param(self, sequence, freq_start, freq_stop, freq_count,  start, stop, count, parameter):

        self.sequence = sequence
        sequence_Z_drive = []
        sequence_Z_readout = []
        sequence_Z_flux = []
        addressed_qubits = []
        starts = []
        durations = []
        self.rel_phases = []
        i = 0
        j = 0
        k = 0
        for pulse in sequence:
            
            qubit = pulse.qubit
            
            if qubit in addressed_qubits:
                pass
            else:
                addressed_qubits.append(qubit)

            starts.append(pulse.start)
            durations.append(pulse.duration)
            self.rel_phases.append(pulse.relative_phase)

            if str(pulse.type) == "PulseType.DRIVE":
                if str(pulse.shape) == "Rectangular()":
                    sequence_Z_drive.append(
                        lo.pulse_library.const(
                            uid=(f"drive_{qubit}_" + str(i)),
                            length=pulse.duration * 1e-9,
                            amplitude=pulse.amplitude,
                        )
                    )
                elif "Gaussian" in str(pulse.shape):
                    sigma = str(pulse.shape).removeprefix("Gaussian(")
                    sigma = float(sigma.removesuffix(")"))
                    sequence_Z_drive.append(
                        lo.pulse_library.gaussian(
                            uid=(f"drive_{qubit}_" + str(i)),
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
                            uid=(f"drive_{qubit}_" + str(i)),
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
                            uid=(f"readout_{qubit}_" + str(j)),
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
            
            if str(pulse.type) == "PulseType.FLUX":
                
                # addressed_qubit.append(pulse.qubit)
                
                if str(pulse.shape) == "Rectangular()":
                    sequence_Z_flux.append(
                        lo.pulse_library.const(
                            uid=(f"flux_{qubit}_" + str(k)),
                            length=pulse.duration * 1e-9,
                            amplitude=pulse.amplitude,
                        )
                    )
            k += 1

        delays = []
        for i in range(len(starts) - 1):
            delays.append(starts[i + 1] - durations[i])
            
        sweep_freq = lo.LinearSweepParameter(uid="freq_sweep", start=freq_start, stop=freq_stop, count=freq_count)
        sweep_parameter = lo.LinearSweepParameter(uid=parameter, start=start, stop=stop, count=count)

        self.SweepFreq = sweep_freq
        self.SweepParameters = sweep_parameter
        self.Parameter = parameter
        # self.SweepParameters = sweep_parameters
        self.delays = delays
        self.sequence_drive = sequence_Z_drive
        self.sequence_readout = sequence_Z_readout
        self.sequence_flux = sequence_Z_flux
        self.addressed_qubits = addressed_qubits
    
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
    
    
    def sequencesPulses_to_exp(self):
        # Create Experiment
        if any(self.sequence_drive) and any(self.sequence_flux) != 0:

            signals = []
            for qubit in self.addressed_qubits:
                signals.append(lo.ExperimentSignal(f"measure{qubit}"))
                signals.append(lo.ExperimentSignal(f"acquire{qubit}"))
                signals.append(lo.ExperimentSignal(f"flux{qubit}"))
                signals.append(lo.ExperimentSignal(f"drive{qubit}"))
            
            exp = lo.Experiment(
                uid="Sequence",
                signals=signals,
            )

            # outer loop - real-time, cyclic averaging in standard integration mode
            with exp.acquire_loop_rt(
                uid="shots",
                count=self.settings["hardware_avg"],
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                averaging_mode=lo.AveragingMode.CYCLIC,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):

                for j in range(len(self.sequences)):
                    sequence_f = self.sequence_flux[j]
                    sequence_d = self.sequence_drive[j]
                    sequence_r = self.sequence_readout[j]
                    sequence_w = self.sequence_weight[j]
                
                    # # inner loop - real-time sweep
                    with exp.section(uid=f"sequence{j}_flux_bias", alignment=lo.SectionAlignment.RIGHT):
                        for pulse in sequence_f:
                            qubit = pulse.uid.split("_")[1]
                            exp.play(signal=f"flux{qubit}", pulse=pulse)

                    with exp.section(uid=f"sequence{j}_drive", alignment=lo.SectionAlignment.RIGHT):
                        i = 0
                        exp.delay(signal=f"drive{qubit}", time = 10e-9) #ramp up
                        for pulse in sequence_d:
                            qubit = pulse.uid.split("_")[1]
                            exp.play(signal=f"drive{qubit}", pulse=pulse, phase=self.rel_phases[j][i])

                            if self.delays[j][i] > 0:
                                qubit = pulse.uid.split("_")[1]
                                exp.delay(signal=f"drive{qubit}", time=self.delays[j][i] * 1e-9)
                            i += 1

                    # qubit readout pulse and data acquisition
                    with exp.section(uid=f"sequence{j}_measure"):
                        for pulse in sequence_r:
                            qubit = pulse.uid.split("_")[1]
                            exp.reserve(signal=f"drive{qubit}")
                            exp.play(signal=f"measure{qubit}", pulse=pulse, phase=self.rel_phases[j][i])
                            exp.acquire(signal=f"acquire{qubit}", handle=f"sequence{j}", kernel=sequence_w[0])

                    # relax time after readout - for signal processing and qubit relaxation to ground state
                    with exp.section(uid=f"sequence{j}_relax", length=self.settings["readout_delay"]):
                        for qubit in self.addressed_qubits:
                            exp.reserve(signal=f"drive{qubit}")
                            exp.reserve(signal=f"flux{qubit}")
                            exp.delay(signal=f"measure{qubit}", time=self.settings["readout_delay"])


        elif any(self.sequence_flux) and not any(self.sequence_drive):
            
            exp = lo.Experiment(
                uid="Sequence",
                signals=[
                    lo.ExperimentSignal("flux"),
                    lo.ExperimentSignal("measure"),
                    lo.ExperimentSignal("acquire"),
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
                
                for j in range(len(self.sequence_readout)):
                    sequence_f = self.sequence_flux[j]
                    sequence_r = self.sequence_readout[j]
                    sequence_w = self.sequence_weight[j]

                # qubit readout pulse and data acquisition

                    with exp.section(uid=f"sequence{j}_flux_bias", alignment=lo.SectionAlignment.RIGHT):
                        for pulse in sequence_f:
                            exp.play(signal="flux", pulse=pulse)

                    # qubit readout pulse and data acquisition
                    with exp.section(uid=f"sequence{j}_measure"):
                        exp.play(signal="measure", pulse=sequence_r[0], phase=self.rel_phases[j][i])
                        exp.acquire(signal="acquire", handle=f"sequence{j}", kernel=sequence_w[0])

                    # relax time after readout - for signal processing and qubit relaxation to ground state
                    with exp.section(uid=f"sequence{j}_relax", length=self.settings["readout_delay"]):
                        exp.delay(signal="measure", time=self.settings["readout_delay"])
                        
        elif any(self.sequence_drive) and not any(self.sequence_flux):
            pass
        
        else:
            
            signals = []
            for qubit in self.addressed_qubits:
                signals.append(lo.ExperimentSignal(f"measure{qubit}"))
                signals.append(lo.ExperimentSignal(f"acquire{qubit}"))
            
            exp = lo.Experiment(
                uid="Sequence",
                signals=signals,
            )

            ## experimental pulse sequence
            # outer loop - real-time, cyclic averaging in spectroscopy integration mode
            with exp.acquire_loop_rt(
                uid="shots",
                count=self.settings["hardware_avg"],
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                averaging_mode=lo.AveragingMode.CYCLIC,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):           
                    
                for j in range(len(self.sequences)):
                    self.iteration = j
                    sequence_r = self.sequence_readout[j]
                    sequence_w = self.sequence_weight[j]
                
                    with exp.section(uid=f"sequence{j}_measure"):
                        i=0
                        for pulse in sequence_r:
                            qubit = pulse.uid.split("_")[1]
                            exp.play(signal=f"measure{qubit}", pulse=pulse, phase=self.rel_phases[j][i])
                            exp.acquire(signal=f"acquire{qubit}", handle=f"sequence{j}", kernel=sequence_w[i])
                            i+=1

                    if self.settings["Fast_reset"] == True:
                        pass
                        # with exp.section(uid=f"sequence{j}_fast_reset", length= ):
                        #     with exp.match_local(handle=f"acquire{qubit}"):
                        #                 with case(state=0):
                        #                     exp.play(some_pulse)
                        #                 with exp.case(state=1):
                        #                     exp.play(some_other_pulse)
                    else:
                        with exp.section(uid=f"sequence{j}_relax", length=self.settings["readout_delay"]):
                            for qubit in self.addressed_qubits:
                                exp.delay(signal=f"measure{qubit}", time=self.settings["readout_delay"])
                                # exp.delay(signal="flux", time=self.settings["readout_delay"])

        

        self.set_maps()
        exp.set_signal_map(self.map_q)

        self.experiment = exp
        
    def sequencePulses_to_exp_Sweeps(self):
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

                with exp.sweep(uid="sweep", parameter=self.SweepParameters, alignment=lo.SectionAlignment.RIGHT):

                    with exp.section(uid="qubit_excitation", alignment=lo.SectionAlignment.RIGHT):
                        i = 0
                        for pulse in self.sequence_drive:

                            if self.Parameter == "Lenght":

                                exp.play(signal="drive", pulse=pulse, length=self.SweepParameters, phase=self.rel_phases[i])

                            if self.Parameter == "Amp":

                                exp.play(
                                    signal="drive", pulse=pulse, amplitude=self.SweepParameters, phase=self.rel_phases[i]
                                )

                            # if self.delays[i] > 0:
                            #     exp.delay(signal="drive", time=self.delays[i] * 1e-9)
                            # i += 1

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

        
        else: 
            
            signals = []
            for qubit in self.addressed_qubits:
                signals.append(lo.ExperimentSignal(f"measure{qubit}"))
                signals.append(lo.ExperimentSignal(f"acquire{qubit}"))
                signals.append(lo.ExperimentSignal(f"flux{qubit}"))
            
            exp = lo.Experiment(
                uid="Sequence",
                signals=signals,
            )

            ## experimental pulse sequence
            # outer loop - real-time, cyclic averaging in standard integration mode

            with exp.acquire_loop_rt(
                uid="shots",
                count=self.settings["hardware_avg"],
                repetition_mode= lo.RepetitionMode.CONSTANT,
                repetition_time= 100e-6,
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                averaging_mode=lo.AveragingMode.CYCLIC,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):
                
                # for j in range(len(self.sequence)):
                #     sequence_f = self.sequence_flux[j]
                #     sequence_r = self.sequence_readout[j]
                    # sequence_w = self.sequence_weight[j]
                
                with exp.sweep(uid="sweep_freq", parameter=self.SweepFreq, alignment=lo.SectionAlignment.RIGHT):
                    with exp.sweep(uid="sweep_param", parameter=self.SweepParameters):
                        with exp.section(uid="flux bias"):
                            for pulse in self.sequence_flux:
                                qubit = pulse.uid.split("_")[1]
                                if self.Parameter == "Amp":
                                    exp.play(signal=f"flux{qubit}", pulse=pulse, amplitude=self.SweepParameters)
                                    
                        # qubit readout pulse and data acquisition
                        with exp.section(uid="qubit_readout"):
                            for pulse in self.sequence_readout:
                                qubit = pulse.uid.split("_")[1]
                                exp.play(signal=f"measure{qubit}", pulse=pulse, length = 1e-6)
                                # integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]
                                # exp.acquire(signal=f"acquire{qubit}", handle=f"sequence", length=integration_time)
                                exp.acquire(signal=f"acquire{qubit}", handle=f"sequence", kernel=self.readout_weighting_function)

                                # exp.acquire(
                                #     signal="acquire",
                                #     handle="Sequence",
                                #     kernel=self.readout_weighting_function,
                                # )
                        # relax time after readout - for signal processing and qubit relaxation to ground state
                        with exp.section(uid="relax"):
                            for qubit in self.addressed_qubits:
                                exp.delay(signal=f"measure{qubit}", time=self.settings["readout_delay"])
                            
                            
                        
                    
         # define experiment calibration - sweep over qubit drive frequency
            exp_calib = lo.Calibration()
            for qubit in self.addressed_qubits:
                exp_calib[f"measure{qubit}"] = lo.SignalCalibration(
                    oscillator=lo.Oscillator(
                        frequency=self.SweepFreq,
                        modulation_type=lo.ModulationType.HARDWARE,
                    )
                )

        exp.set_calibration(exp_calib)

        self.set_maps()
        exp.set_signal_map(self.map_q)
        self.experiment = exp

    def sequencePulses_to_exp_freqs(self, start, stop, points):
        # Create Experiment

        frequency_sweep = lo.LinearSweepParameter(uid=f"frequency_sweep", start=start, stop=stop, count=points)

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
                count=self.settings["hardware_avg"],
                averaging_mode=lo.AveragingMode.SEQUENTIAL,
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                # averaging_mode=lo.AveragingMode.CYCLIC,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):

                with exp.sweep(parameter=frequency_sweep):
                    j = 0
                    # # inner loop - real-time sweep of qubit drive pulse amplitude
                    # with exp.sweep(uid="sweep", parameter=sweep_rel_flat, alignment=SectionAlignment.RIGHT):
                    # qubit excitation - pulse amplitude will be swept
                    with exp.section(alignment=lo.SectionAlignment.RIGHT):
                        i = 0
                        for pulse in self.sequence_drive:
                            exp.play(signal="drive", pulse=pulse, phase=self.rel_phases[i])

                            if self.delays[i] > 0:
                                exp.delay(signal="drive", time=self.delays[i] * 1e-9)
                            i += 1

                        # exp.delay(signal="drive", time=300e-9)

                    # qubit readout pulse and data acquisition
                    with exp.section():
                        for pulse in self.sequence_readout:

                            exp.reserve(signal="drive")
                            exp.play(signal="measure", pulse=pulse, phase=self.rel_phases[i])

                            integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]

                            exp.acquire(signal="acquire", handle=f"sequence{j}", length=integration_time)

                            j += 1

                    # relax time after readout - for signal processing and qubit relaxation to ground state
                    with exp.section():
                        exp.reserve(signal="drive")
                        exp.delay(signal="measure", time=self.settings["readout_delay"])

            # define experiment calibration - sweep over qubit drive frequency
            exp_calib = lo.Calibration()
            exp_calib["drive"] = lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=frequency_sweep,
                    modulation_type=lo.ModulationType.HARDWARE,
                )
            )

        # TODO: Add features of above to else
        else:
            
            qubits = self.addressed_qubit
            self.addressed_qubits = self.addressed_qubit
            
            signals = []
            for qubit in qubits:
                signals.append(lo.ExperimentSignal(f"measure{qubit}"))
                signals.append(lo.ExperimentSignal(f"acquire{qubit}"))
            
            exp = lo.Experiment(
                uid="Sequence",
                signals=signals,
            )

            ## experimental pulse sequence
            # outer loop - real-time, cyclic averaging in standard integration mode
            with exp.acquire_loop_rt(
                count=self.settings["hardware_avg"],
                averaging_mode=lo.AveragingMode.CYCLIC,
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):
                # # inner loop - real-time sweep of qubit drive pulse amplitude
                # qubit readout pulse and data acquisition
                with exp.sweep(parameter=frequency_sweep):
                    j = 0
                    with exp.section(alignment=lo.SectionAlignment.RIGHT):
                        i = 0
                        for qubit in qubits:
                            for pulse in self.sequence_readout:
                                exp.play(signal=f"measure{qubit}", pulse=pulse, phase=self.rel_phases[i])
                                integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]
                                exp.acquire(
                                    signal=f"acquire{qubit}",
                                    handle=f"sequence{j}",
                                    length=integration_time,
                                )
                            j += 1

                    # relax time after readout - for signal processing and qubit relaxation to ground state
                    with exp.section():
                        exp.delay(signal=f"measure{qubit}", time=self.settings["readout_delay"])

            # define experiment calibration - sweep over qubit drive frequency
            exp_calib = lo.Calibration()
            exp_calib[f"measure{qubit}"] = lo.SignalCalibration(
                oscillator=lo.Oscillator(
                    frequency=frequency_sweep,
                    modulation_type=lo.ModulationType.HARDWARE,
                )
            )

        exp.set_calibration(exp_calib)

        self.set_maps()
        exp.set_signal_map(self.map_q)
        self.experiment = exp
        
    def sequencePulses_to_exp_freqs_multi(self, qubits):
        # Create Experiment

        # qubits = self.addressed_qubit
        
        signals = []
        for qubit in qubits:
            signals.append(lo.ExperimentSignal(f"measure{qubit}"))
            signals.append(lo.ExperimentSignal(f"acquire{qubit}"))
        
        exp = lo.Experiment(
            uid="Sequence",
            signals=signals,
        )

        ## experimental pulse sequence
        # outer loop - real-time, cyclic averaging in standard integration mode
        with exp.acquire_loop_rt(
            count=self.settings["hardware_avg"],
            averaging_mode=lo.AveragingMode.CYCLIC,
            # acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
            acquisition_type=lo.AcquisitionType.INTEGRATION,
        ):
            # # inner loop - real-time sweep of qubit drive pulse amplitude
            # qubit readout pulse and data acquisition
            
            with exp.section(alignment=lo.SectionAlignment.RIGHT):
                i = 0
                for pulse in self.sequence_readout:
                    exp.play(signal=f"measure0", pulse=pulse, phase=self.rel_phases[i])
                    exp.play(signal=f"measure1", pulse=pulse, phase=self.rel_phases[i])
                    integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]
                    exp.acquire(
                        signal=f"acquire0",
                        handle=f"sequence0",
                        length=integration_time,
                    )
                    
                    exp.acquire(
                        signal=f"acquire1",
                        handle=f"sequence1",
                        length=integration_time,
                    )
                    
                    i += 1
                
                # for qubit in qubits:
                #     for pulse in self.sequence_readout:
                #         exp.play(signal=f"measure{qubit}", pulse=pulse, phase=self.rel_phases[i])
                #         integration_time = self.native_gates["single_qubit"][0]["MZ"]["integration_time"]
                #         exp.acquire(
                #             signal=f"acquire{qubit}",
                #             handle=f"sequence{j}",
                #             length=integration_time,
                #         )
                #     j += 1

            # relax time after readout - for signal processing and qubit relaxation to ground state
            with exp.section():
                exp.delay(signal=f"measure{qubit}", time=self.settings["readout_delay"])

        self.set_maps(qubits)
        exp.set_signal_map(self.map_q)
        self.experiment = exp
        

        # if self.sequence == sequence:
        #     self.repeat_seq()
        # else:
        #     self.sequence_to_ZurichPulses(sequence)
        #     self.sequencePulses_to_exp()
        #     self.run_seq()

        self.sequence_to_ZurichPulses(sequence)
        self.sequencePulses_to_exp()
        self.run_seq()
        
        spec_res = self.results.get_data("sequence")

        msr = abs(spec_res)
        # phase = np.unwrap(np.angle(spec_res))
        phase = np.angle(spec_res)
        i = spec_res.real
        q = spec_res.imag

        return msr, phase, i, q
    
    
    
    
    
    def execute_flux_sequences(self, sequences):

        # if self.sequence == sequence:
        #     self.repeat_seq()
        # else:
        #     self.sequence_to_ZurichPulses(sequence)
        #     self.sequencePulses_to_exp()
        #     self.run_seq()

        self.sequences_to_ZurichPulses(sequences)
        self.sequencesPulses_to_exp()
        self.run_seq()
        
        spec_res = []
        msr = []
        phase = []
        i = []
        q = []
        for j in range(len(self.sequences)):
            spec_res.append(self.results.get_data(f"sequence{j}"))
            msr.append(abs(spec_res[j]))
            phase.append(np.angle(spec_res[j]))
            i.append(spec_res[j].real)
            q.append(spec_res[j].imag)

        return msr, phase, i, q
    
    def execute_sequences(self, sequences, sweepers = None):

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
        for j in range(len(self.sequences)):
            spec_res.append(self.results.get_data(f"sequence{j}"))
            msr.append(abs(spec_res[j]))
            phase.append(np.angle(spec_res[j]))
            i.append(spec_res[j].real)
            q.append(spec_res[j].imag)

        return msr, phase, i, q
    
    def execute_flux_sequence_freq_param(self, sequence, freq_start, freq_stop, freq_count, start, stop, count, parameter):

        self.sequence_to_ZurichSweep_freq_param(sequence, freq_start, freq_stop, freq_count,  start, stop, count, parameter)
        self.sequencePulses_to_exp_Sweeps()
        self.run_seq()

        spec_res = []
        msr = []
        phase = []
        i = []
        q = []
        for j in range(freq_count):
            for k in range(count):
                datapoint = self.results.get_data("sequence")[j][k]
                msr.append(abs(datapoint))
                phase.append(np.angle(datapoint))
                i.append(datapoint.real)
                q.append(datapoint.imag)

        return msr, phase, i, q
    
    def execute_pulse_sequence_freq(self, sequence, start, stop, points):

        self.sequence_to_ZurichPulses(sequence)
        self.sequencePulses_to_exp_freqs(start, stop, points)
        self.run_seq()

        spec_res = []
        msr = []
        phase = []
        i = []
        q = []
        for j in range(len(self.sequence_readout)):
            spec_res.append(self.results.get_data(f"sequence{j}"))
            msr.append(abs(spec_res[j]))
            phase.append(np.angle(spec_res[j]))
            i.append(spec_res[j].real)
            q.append(spec_res[j].imag)

        return msr, phase, i, q
    
    def execute_pulse_sequence_freq_multi(self, sequence, qubits):

        # if self.sequence == sequence:
        #     self.repeat_seq()
        # else:
        #     self.sequence_to_ZurichPulses(sequence)
        #     self.sequencePulses_to_exp()
        #     self.run_seq()

        self.sequence_to_ZurichPulses(sequence)
        self.sequencePulses_to_exp_freqs_multi(qubits)
        self.run_multi()

        spec_res = []
        msr = []
        phase = []
        i = []
        q = []
        for k in range(len(qubits)):
            spec_res.append(self.results.get_data(f"sequence{k}"))
            msr.append(abs(spec_res[k]))
            phase.append(np.angle(spec_res[k]))
            i.append(spec_res[k].real)
            q.append(spec_res[k].imag)

        return msr, phase, i, q
    
    
    
    
    
    
    #TODO:ERASE
    def sequence_to_ZurichPulses(self, sequence):
        self.sequence = sequence
        sequence_Z_drive = []
        sequence_Z_readout = []
        sequence_Z_flux = []
        starts = []
        durations = []
        addressed_qubit = []
        self.rel_phases = []
        i = 0
        j = 0
        k = 0
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
                addressed_qubit.append(pulse.qubit)
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
            
            if str(pulse.type) == "PulseType.FLUX":
                if str(pulse.shape) == "Rectangular()":
                    sequence_Z_flux.append(
                        lo.pulse_library.const(
                            uid=("flux" + str(k)),
                            length=pulse.duration * 1e-9,
                            amplitude=pulse.amplitude,
                        )
                    )
            k += 1

        delays = []
        for i in range(len(starts) - 1):
            delays.append(starts[i + 1] - durations[i])

        self.delays = delays
        self.sequence_drive = sequence_Z_drive
        self.sequence_readout = sequence_Z_readout
        self.sequence_flux = sequence_Z_flux
        self.addressed_qubit = addressed_qubit
        
    def sequencePulses_to_exp(self):
        # Create Experiment

        if len(self.sequence_drive) != 0:
            exp = lo.Experiment(
                uid="Sequence",
                signals=[
                    lo.ExperimentSignal("flux"),
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
                with exp.section(uid="flux bias"):
                    for pulse in self.sequence_flux:
                        exp.play(signal="flux", pulse=pulse)
                
            #     with exp.sweep(uid="flux_sweep", parameter=flux_sweep):
            # with exp.section(uid="flux bias"):
            #     exp.play(signal="flux", pulse=const_flux, amplitude=flux_sweep)
                
                with exp.section(uid="qubit_excitation", alignment=lo.SectionAlignment.RIGHT):
                    i = 0
                    exp.delay(signal="drive", time = 10e-9)
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
                    lo.ExperimentSignal("flux"),
                    lo.ExperimentSignal("measure"),
                    lo.ExperimentSignal("acquire"),
                ],
            )
            ## experimental pulse sequence
            # outer loop - real-time, cyclic averaging in standard integration mode
            with exp.acquire_loop_rt(
                uid="shots",
                count=self.settings["hardware_avg"],
                averaging_mode=lo.AveragingMode.CYCLIC,
                # averaging_mode=lo.AveragingMode.SEQUENTIAL,
                acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
                # acquisition_type=lo.AcquisitionType.INTEGRATION,
            ):
                # # inner loop - real-time sweep of qubit drive pulse amplitude
                # qubit readout pulse and data acquisition

                with exp.section(uid="flux bias"):
                    for pulse in self.sequence_flux:
                        exp.play(signal="flux", pulse=pulse)
                        
                i = 0
                with exp.section(uid="qubit_readout"):
                    for pulse in self.sequence_readout:
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

        self.set_maps(self.addressed_qubit)
        exp.set_signal_map(self.map_q)

        self.experiment = exp
    
    def execute_flux_sequence(self, sequence):
        # if self.sequence == sequence:
        #     self.repeat_seq()
        # else:
        #     self.sequence_to_ZurichPulses(sequence)
        #     self.sequencePulses_to_exp()
        #     self.run_seq()

        self.sequence_to_ZurichPulses(sequence)
        self.sequencePulses_to_exp()
        self.run_seq()
        
        spec_res = []
        msr = []
        phase = []
        i = []
        q = []

        spec_res.append(self.results.get_data(f"sequence"))
        msr.append(abs(spec_res[j]))
        phase.append(np.angle(spec_res[j]))
        i.append(spec_res[j].real)
        q.append(spec_res[j].imag)

        return msr, phase, i, q
    
    def create_Measure_exp(self):
        signals = []
        for qubit in self.addressed_qubits:
            signals.append(lo.ExperimentSignal(f"measure{qubit}"))
            signals.append(lo.ExperimentSignal(f"acquire{qubit}"))
        
        exp = lo.Experiment(
            uid="Sequence",
            signals=signals,
        )

        with exp.acquire_loop_rt(
            uid="shots",
            count=self.settings["hardware_avg"],
            acquisition_type=lo.AcquisitionType.SPECTROSCOPY,
            averaging_mode=lo.AveragingMode.CYCLIC,
            # acquisition_type=lo.AcquisitionType.INTEGRATION,
        ):

            for j in range(len(self.sequences)):
                self.iteration = j
                # exp = self.Measure(exp)
                # exp = self.qubit_reset(exp)
                self.Measure(exp)
                self.qubit_reset(exp)

            
        self.exp = exp
        
        self.set_maps()
        exp.set_signal_map(self.map_q)

        self.experiment = exp

    def Measure_sequences(self, sequences):

        # if self.sequence == sequence:
        #     self.repeat_seq()
        # else:
        #     self.sequence_to_ZurichPulses(sequence)
        #     self.sequencePulses_to_exp()
        #     self.run_seq()

        self.sequences_to_ZurichPulses(sequences)
        self.create_Measure_exp()
        self.run_seq()
        
        spec_res = []
        msr = []
        phase = []
        i = []
        q = []
        for j in range(len(self.sequences)):
            spec_res.append(self.results.get_data(f"sequence{j}"))
            msr.append(abs(spec_res[j]))
            phase.append(np.angle(spec_res[j]))
            i.append(spec_res[j].real)
            q.append(spec_res[j].imag)

        return msr, phase, i, q
    
    
    # TODO: Add more gates
    def create_qubit_readout_pulse(self, qubit, start):
        ro_duration = self.native_gates["single_qubit"][qubit]["MZ"]["duration"]
        ro_frequency = self.native_gates["single_qubit"][qubit]["MZ"]["frequency"]
        ro_amplitude = self.native_gates["single_qubit"][qubit]["MZ"]["amplitude"]
        ro_shape = self.native_gates["single_qubit"][qubit]["MZ"]["shape"]
        ro_channel = self.qubit_channel_map[qubit][0]
        from qibolab.pulses import ReadoutPulse

        return ReadoutPulse(start, ro_duration, ro_amplitude, ro_frequency, 0, ro_shape, ro_channel, qubit=qubit)

    def create_qubit_flux_pulse(self, qubit, start, duration):
        flux_duration = duration
        flux_amplitude = self.native_gates["single_qubit"][qubit]["Flux"]["amplitude"]
        flux_shape = self.native_gates["single_qubit"][qubit]["Flux"]["shape"]
        flux_channel = self.qubit_channel_map[qubit][0]
        from qibolab.pulses import FluxPulse

        return FluxPulse(start, flux_duration, flux_amplitude, 0, flux_shape, flux_channel, qubit=qubit)
    
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