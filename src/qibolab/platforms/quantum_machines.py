from types import SimpleNamespace

import numpy as np
import yaml
from qibo.config import log, raise_error
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager

from qibolab.instruments.rohde_schwarz import SGS100A
from qibolab.platforms.abstract import AbstractPlatform


class Channel:
    """Representation of physical wire connection (channel).

    Name is used as a unique identifier for channels. If a channel
    with an existing name is recreated, it will refer to the existing object.
    Channel objects are created and their attributes are set during
    the platform instantiation.

    Args:
        name (str): Name of the channel as given in the platform runcard.

    Attributes:
        ports (list): List of tuples (controller (`str`), port (`int`))
            specifying the QM (I, Q) ports that the channel is connected.
        qubits (list): List of Qubit objects for the qubits connected to this channel.
        local_oscillator (:class:`qibolab.instruments.rohde_schwarz.SGS100A):
            Instrument object for the local oscillator connected to this channel.
        time_of_flight (optional,int): Time of flight associated with the channel.
            Relevant only for readout and used for hardware signal integration.
        smearing (optional,int): Time of flight associated with the channel.
            Relevant only for readout and used for hardware signal integration.
    """

    # TODO: Maybe this dictionary can be moved inside platform
    instances = {}

    def __new__(cls, name):
        if name is None:
            return None

        if name not in cls.instances:
            new = super().__new__(cls)
            new.name = name

            new.ports = []
            new.qubits = []
            new.local_oscillator = None

            new.time_of_flight = None
            new.smearing = None

            cls.instances[name] = new

        return cls.instances[name]

    def __repr__(self):
        return f"<Channel {self.name}>"

    def set_lo_frequency(self, frequency):
        """Sets the local oscillator frequency for all qubits connected to the channel.

        This updates the LO frequencies in the QM config.

        Args:
            frequency (int): Frequency of the local oscillator in Hz.
        """
        for qubit, mode in self.qubits:
            getattr(qubit, f"set_{mode}_lo_frequency")(frequency)


class Qubit:
    """Representation of a physical qubit.

    Qubit objects are instantiated during the platform initialization and
    are used to register elements in the QM config.

    Args:
        name (int): Qubit number.
        characterization (dict): Dictionary with the characterization values
            for the qubit, loaded from the runcard.
        drive (:class:`qibolab.platforms.quantum_machines.Channel`): Channel
            used to send drive pulses to the qubit.
        readout (:class:`qibolab.platforms.quantum_machines.Channel`): Channel
            used to send readout pulses to the qubit.
        feedback (:class:`qibolab.platforms.quantum_machines.Channel`): Channel
            used to get readout feedback from the qubit.
        flux (:class:`qibolab.platforms.quantum_machines.Channel`): Channel
            used to send flux pulses to the qubit.
    """

    def __init__(self, name, characterization, drive, readout, feedback, flux=None):
        self.name = name
        self.characterization = SimpleNamespace(**characterization)

        self.drive = drive
        self.readout = readout
        self.feedback = feedback
        self.flux = flux

        # elements entry for ``QuantumMachinesManager`` config
        self.elements = {}
        if drive:
            # update qubits list in the channel
            drive.qubits.append((self, "drive"))
            self.elements[f"drive{self.name}"] = {
                "mixInputs": {
                    "I": self.drive.ports[0],
                    "Q": self.drive.ports[1],
                    "lo_frequency": None,
                    "mixer": f"mixer_drive{self.name}",
                },
                "intermediate_frequency": 0,
                "operations": {},
            }
        if readout:
            # update qubits list in the channel
            readout.qubits.append((self, "readout"))
            self.elements[f"readout{self.name}"] = {
                "mixInputs": {
                    "I": self.readout.ports[0],
                    "Q": self.readout.ports[1],
                    "lo_frequency": None,
                    "mixer": f"mixer_readout{self}",
                },
                "intermediate_frequency": 0,
                "operations": {},
                "outputs": {
                    "out1": self.feedback.ports[0],
                    "out2": self.feedback.ports[1],
                },
                "time_of_flight": self.feedback.time_of_flight,
                "smearing": self.feedback.smearing,
            }
        if flux:
            # update qubits list in the channel
            flux.qubits.append((self, "flux"))
            self.elements[f"flux{self.name}"] = {
                "singleInput": {
                    "port": self.flux.ports[0],
                },
                "intermediate_frequency": 0,
                "operations": {},
            }

        # mixers entry for ``QuantumMachinesManager`` config
        self.mixers = {}
        if drive:
            drive_g = self.characterization.mixer_drive_g
            drive_phi = self.characterization.mixer_drive_phi
            self.mixers[f"mixer_drive{self}"] = [
                {
                    "intermediate_frequency": 0,
                    "lo_frequency": None,
                    "correction": self.iq_imbalance(drive_g, drive_phi),
                }
            ]
        if readout:
            readout_g = self.characterization.mixer_readout_g
            readout_phi = self.characterization.mixer_readout_phi
            self.mixers[f"mixer_readout{self}"] = [
                {
                    "intermediate_frequency": 0,
                    "lo_frequency": None,
                    "correction": self.iq_imbalance(readout_g, readout_phi),
                }
            ]

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return f"<Qubit {self.name}>"

    @staticmethod
    def iq_imbalance(g, phi):
        """Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances

        More information here:
        https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer

        Args:
            g (float): relative gain imbalance between the I & Q ports (unit-less).
                Set to 0 for no gain imbalance.
            phi (float): relative phase imbalance between the I & Q ports (radians).
                Set to 0 for no phase imbalance.
        """
        c = np.cos(phi)
        s = np.sin(phi)
        N = 1 / ((1 - g**2) * (2 * c**2 - 1))
        return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]

    def set_drive_lo_frequency(self, frequency):
        """Updates drive local oscillator frequency in the QM config."""
        # TODO: Maybe this can be moved to ``Channel``
        self.elements[f"drive{self}"]["mixInputs"]["lo_frequency"] = frequency
        self.mixers[f"mixer_drive{self}"][0]["lo_frequency"] = frequency

    def set_readout_lo_frequency(self, frequency):
        """Updates readout local oscillator frequency in the QM config."""
        # TODO: Maybe this can be moved to ``Channel``
        self.elements[f"readout{self}"]["mixInputs"]["lo_frequency"] = frequency
        self.mixers[f"mixer_readout{self}"][0]["lo_frequency"] = frequency

    def set_drive_if_frequency(self, frequency):
        """Updates drive intermediate frequency in the QM config."""
        self.elements[f"drive{self}"]["intermediate_frequency"] = frequency
        self.mixers[f"mixer_drive{self}"][0]["intermediate_frequency"] = frequency

    def set_readout_if_frequency(self, frequency):
        """Updates readout local oscillator frequency in the QM config."""
        self.elements[f"readout{self}"]["intermediate_frequency"] = frequency
        self.mixers[f"mixer_readout{self}"][0]["intermediate_frequency"] = frequency

    def set_flux_frequency(self, frequency):
        """Updates flux pulse frequency in the QM config."""
        self.elements[f"flux{self}"]["intermediate_frequency"] = frequency

    def register_drive_pulse(self, pulse):
        """Registers drive pulse as an operation in the QM config."""
        self.elements[f"drive{self}"]["operations"][pulse.serial] = pulse.serial

    def register_readout_pulse(self, pulse):
        """Registers readout pulse as an operation in the QM config."""
        self.elements[f"readout{self}"]["operations"][pulse.serial] = pulse.serial

    def register_flux_pulse(self, pulse):
        """Registers flux pulse as an operation in the QM config."""
        self.elements[f"flux{self}"]["operations"][pulse.serial] = pulse.serial


class QuantumMachinesPlatform(AbstractPlatform):
    """Platform controlling Quantum Machines (QM) OPX controllers and Rohde Schwarz local oscillators.

    Playing pulses on QM controllers requires a ``config`` dictionary and a program
    written in QUA language. The ``config`` file is generated in parts in the following places:
        - controllers are registered in ``__init__``,
        - elements (qubits, resonators and flux) are registered in each ``Qubit`` object,
        - pulses (including waveforms and integration weights) are registered in the
          ``register_*`` methods of the platform.
    The QUA program for executing an arbitrary qibolab ``PulseSequence`` is written in
    ``execute_pulse_sequence``.

    Args:
        name (str): Name of the platform.
        runcard (str): Path to the runcard.

    The platform has multiple attributes that are loaded from the runcard and processed.

    Attributes:
        is_connected (bool): Boolean that shows whether instruments are connected.
        nqubits (int): Number of qubits in the chip.
        resonator_type (str): Type of the resonators (2D or 3D) used for qubit state readout.
        topology (list): Topology of the chip.
        settings (dict): Raw runcard loaded in a dictionary.
        manager (:class:`qm.QuantumMachinesManager.QuantumMachinesManager`): Manager object
            used for controlling the QM OPXs.
        config (dict): Configuration dictionary required for pulse execution on the OPXs.
        local_oscillators (dict): Dictionary mapping LO names to the corresponding
            instrument objects.
        qubits (list): List of :class:`qibolab.platforms.quantum_machines.Qubit` objects
            for each qubit to be controlled.
    """

    def __init__(self, name, runcard):
        log.info(f"Loading platform {name} from runcard {runcard}")
        self.name = name
        self.runcard = runcard
        self.is_connected = False
        # Load platform settings
        with open(runcard) as file:
            self.settings = yaml.safe_load(file)

        self.nqubits = self.settings["nqubits"]
        self.resonator_type = "3D" if self.nqubits == 1 else "2D"
        self.topology = self.settings["topology"]

        # QuantumMachines manager is instantiated in ``platform.connect``
        self.manager = None

        settings = self.settings["settings"]

        # Create feedback channel (for readout input to instrument)
        feedback = Channel(self.settings["feedback_channel"])
        feedback.time_of_flight = settings["time_of_flight"]
        feedback.smearing = settings["smearing"]

        # Default configuration for communicating with the ``QuantumMachinesManager``
        self.config = {
            "version": 1,
            "controllers": {},
            "elements": {},
            "pulses": {},
            "waveforms": {},
            "digital_waveforms": {
                "ON": {"samples": [(1, 0)]},
            },
            "integration_weights": {},
            "mixers": {},
        }

        instruments = dict(self.settings["instruments"])
        qm_settings = instruments.pop("qm")
        # Register controllers in config and channels
        for controller, values in qm_settings["controllers"].items():
            self.config["controllers"][controller] = values["ports"]
            for channel_name, ports in values["channel_port_map"].items():
                Channel(channel_name).ports = [(controller, p) for p in ports]

        # Instantiate local oscillators
        self.local_oscillators = {}
        for name, value in instruments.items():
            lo = SGS100A(name, value["address"])
            self.local_oscillators[name] = lo
            for channel_name in value["channels"]:
                Channel(channel_name).local_oscillator = lo

        # Create list of qubit objects
        characterization = self.settings["characterization"]["single_qubit"]
        self.qubits = []
        for q, channel_names in self.settings["qubit_channel_map"].items():
            readout, drive, flux = (Channel(name) for name in channel_names)
            self.qubits.append(Qubit(q, characterization[q], drive, readout, feedback, flux))

        self.reload_settings()

    def reload_settings(self):
        """Reloads the runcard and re-setups the connected instruments using the new values."""
        with open(self.runcard) as file:
            self.settings = yaml.safe_load(file)

        if self.is_connected:
            self.setup()

    def run_calibration(self, show_plots=False):  # pragma: no cover
        raise_error(NotImplementedError)

    def connect(self, host=None):
        """Connects to all instruments.

        Args:
            host (optional, str): Optional can be given to connect to a cloud simulator
                instead of the actual instruments. If the ``host`` is not given,
                this will connect to the physical OPXs and LOs using the addresses
                given in the runcard.
        """
        if host:
            from qm.simulate.credentials import create_credentials

            host, port = host.split(":")
            self.manager = QuantumMachinesManager(host, int(port), credentials=create_credentials())
        else:
            host, port = self.settings["instruments"]["qm"]["address"].split(":")
            self.manager = QuantumMachinesManager(host, int(port))
            if not self.is_connected:
                try:
                    for name in self.local_oscillators:
                        log.info(f"Connecting to {self.name} instrument {name}.")
                        self.local_oscillators[name].connect()
                    self.is_connected = True
                except Exception as exception:
                    raise_error(
                        RuntimeError,
                        f"Cannot establish connection to {self.name} instruments. Error captured: '{exception}'",
                    )

    def setup(self):
        for name, lo in self.local_oscillators.items():
            inst = self.settings["instruments"][name]
            if self.is_connected:
                lo.setup(**inst["settings"])
            else:
                log.warn(f"There is no connection to {name}. Frequencies were not set.")
            frequency = inst["settings"]["frequency"]
            for channel_name in inst["channels"]:
                Channel(channel_name).set_lo_frequency(frequency)

    def start(self):
        # TODO: Start the OPX flux offsets?
        if self.is_connected:
            for lo in self.local_oscillators.values():
                lo.start()

    def stop(self):
        if self.is_connected:
            for lo in self.local_oscillators.values():
                lo.stop()
            self.manager.close_all_quantum_machines()

    def disconnect(self):
        if self.is_connected:
            for lo in self.local_oscillators.values():
                lo.disconnect()
            self.manager.close()
            self.is_connected = False

    def register_waveform(self, pulse, mode="i"):
        """Registers waveforms in QM config.

        QM supports two kinds of waveforms, examples:
            "zero_wf": {"type": "constant", "sample": 0.0}
            "x90_wf": {"type": "arbitrary", "samples": x90_wf.tolist()}

        Args:
            pulse (:class:`qibolab.pulses.Pulse`): Pulse object to read the waveform from.
            mode (str): "i" or "q" specifying which channel the waveform will be played.

        Returns:
            serial (str): String with a serialization of the waveform.
                Used as key to identify the waveform in the config.
        """
        from qibolab.pulses import Rectangular

        waveforms = self.config["waveforms"]
        if isinstance(pulse.shape, Rectangular):
            serial = f"constant_wf{pulse.amplitude}"
            if serial not in waveforms:
                waveforms[serial] = {"type": "constant", "sample": pulse.amplitude}
        else:
            waveform = getattr(pulse, f"envelope_waveform_{mode}")
            serial = waveform.serial
            if serial not in waveforms:
                waveforms[serial] = {"type": "arbitrary", "samples": waveform.data.tolist()}
        return serial

    def register_integration_weights(self, qubit, readout_len):
        """Registers integration weights in QM config.

        Args:
            qubit (:class:`qibolab.platforms.quantum_machines.Qubit`): Qubit
                object that the integration weights will be used for.
            readout_len (int): Duration of the readout pulse in ns.
        """
        rotation_angle = qubit.characterization.rotation_angle
        self.config["integration_weights"] = {
            f"cosine_weights{qubit}": {
                "cosine": [(np.cos(rotation_angle), readout_len)],
                "sine": [(-np.sin(rotation_angle), readout_len)],
            },
            f"sine_weights{qubit}": {
                "cosine": [(np.sin(rotation_angle), readout_len)],
                "sine": [(np.cos(rotation_angle), readout_len)],
            },
            f"minus_sine_weights{qubit}": {
                "cosine": [(-np.sin(rotation_angle), readout_len)],
                "sine": [(-np.cos(rotation_angle), readout_len)],
            },
        }

    def register_pulse(self, pulse):
        """Registers pulse, waveforms and integration weights in QM config.

        Args:
            pulse (:class:`qibolab.pulses.Pulse`): Pulse object to register.

        Returns:
            element (str): Name of the element this pulse will be played on.
                Elements are a part of the QM config and are generated during
                instantiation of the Qubit objects. They are named as
                "drive0", "drive1", "flux0", "readout0", ...
        """
        pulses = self.config["pulses"]
        qubit = self.qubits[pulse.qubit]
        if pulse.serial not in pulses:
            # upload qubit elements and mixers to Quantum Machines config
            self.config["elements"].update(qubit.elements)
            self.config["mixers"].update(qubit.mixers)

            if pulse.type.name == "DRIVE":
                serial_i = self.register_waveform(pulse, "i")
                serial_q = self.register_waveform(pulse, "q")
                pulses[pulse.serial] = {
                    "operation": "control",
                    "length": pulse.duration,
                    "waveforms": {"I": serial_i, "Q": serial_q},
                }
                qubit.set_drive_if_frequency(pulse.frequency)
                qubit.register_drive_pulse(pulse)

            elif pulse.type.name == "FLUX":
                serial = self.register_waveform(pulse)
                self.qubits[pulse.qubit].set_flux_frequency(pulse.frequency)
                pulses[pulse.serial] = {
                    "operation": "control",
                    "length": pulse.duration,
                    "waveforms": {
                        "single": serial,
                    },
                }
                qubit.set_flux_frequency(pulse.frequency)
                qubit.register_flux_pulse(pulse)

            elif pulse.type.name == "READOUT":
                serial_i = self.register_waveform(pulse, "i")
                serial_q = self.register_waveform(pulse, "q")
                self.register_integration_weights(qubit, pulse.duration)
                pulses[pulse.serial] = {
                    "operation": "measurement",
                    "length": pulse.duration,
                    "waveforms": {
                        "I": serial_i,
                        "Q": serial_q,
                    },
                    "integration_weights": {
                        "cos": f"cosine_weights{pulse.qubit}",
                        "sin": f"sine_weights{pulse.qubit}",
                        "minus_sin": f"minus_sine_weights{pulse.qubit}",
                    },
                    "digital_marker": "ON",
                }
                qubit.set_readout_if_frequency(pulse.frequency)
                qubit.register_readout_pulse(pulse)

            else:
                raise_error(TypeError, f"Unknown pulse type {pulse.type.name}.")

        return f"{pulse.type.name.lower()}{str(qubit)}"

    def execute_program(self, program, simulation_duration=None):
        """Executes an arbitrary program written in QUA language.

        Args:
            program: QUA program.
            simulation_duration (optional, int): Duration for the simulation in ns.
                If not given the program will be executed using the real instruments.

        Returns:
            TODO
        """
        if simulation_duration:
            # controller_connections = create_simulator_controller_connections(3)
            simulation_config = SimulationConfig(duration=simulation_duration // 4)
            return self.manager.simulate(self.config, program, simulation_config)
        else:
            machine = self.manager.open_qm(self.config)
            return machine.execute(program)

    def execute_pulse_sequence(self, sequence, nshots=None, simulation_duration=None):
        """Executes an arbitrary pulse sequence.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to play.
            nshots (int): Number of (hardware) repetitions for the execution.
            simulation_duration (optional, int): Duration for the simulation in ns.
                If not given the pulse execution will be played using real instruments.

        Returns:
            TODO
        """
        from qm.qua import (
            declare,
            declare_stream,
            dual_demod,
            fixed,
            for_,
            measure,
            play,
            program,
            save,
            stream_processing,
            wait,
        )

        if nshots is None:
            nshots = self.settings["settings"]["hardware_avg"]

        # TODO: Fix phases
        # TODO: Handle pulses that run on the same element simultaneously (multiplex?)
        # register pulses in Quantum Machines config
        targets = [self.register_pulse(pulse) for pulse in sequence]
        # play pulses using QUA
        clock = {target: 0 for target in targets}
        with program() as experiment:
            n = declare(int)
            I = declare(fixed)
            Q = declare(fixed)
            I_st = declare_stream()
            Q_st = declare_stream()
            with for_(n, 0, n < nshots, n + 1):
                for pulse, target in zip(sequence, targets):
                    wait_time = pulse.start - clock[target]
                    if wait_time > 0:
                        wait(wait_time // 4, target)
                    clock[target] += pulse.duration
                    if pulse.type.name == "READOUT":
                        # align("qubit", "resonator")
                        measure(
                            pulse.serial,
                            target,
                            None,
                            dual_demod.full("cos", "out1", "sin", "out2", I),
                            dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                        )
                    else:
                        play(pulse.serial, target)
                # Save data to the stream processing
                save(I, I_st)
                save(Q, Q_st)

            with stream_processing():
                # I_st.average().save("I")
                # Q_st.average().save("Q")
                # n_st.buffer().save_all("n")
                I_st.buffer(nshots).save("I")
                Q_st.buffer(nshots).save("Q")

        return self.execute_program(experiment, simulation_duration)
