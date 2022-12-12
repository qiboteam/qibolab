from types import SimpleNamespace

import numpy as np
import yaml
from qibo.config import log, raise_error
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager

from qibolab.instruments.rohde_schwarz import SGS100A
from qibolab.platforms.abstract import AbstractPlatform


def iq_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer
    :param g: relative gain imbalance between the I & Q ports (unit-less). Set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the I & Q ports (radians). Set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


class Channel:

    # TODO: Move this dictionary inside the platform
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
        for qubit, mode in self.qubits:
            getattr(qubit, f"set_{mode}_lo_frequency")(frequency)


class Qubit:
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
                    "correction": iq_imbalance(drive_g, drive_phi),
                }
            ]
        if readout:
            readout_g = self.characterization.mixer_readout_g
            readout_phi = self.characterization.mixer_readout_phi
            self.mixers[f"mixer_readout{self}"] = [
                {
                    "intermediate_frequency": 0,
                    "lo_frequency": None,
                    "correction": iq_imbalance(readout_g, readout_phi),
                }
            ]

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return f"<Qubit {self.name}>"

    def set_drive_lo_frequency(self, frequency):
        # TODO: Maybe this can be moved to ``Channel``
        self.elements[f"drive{self}"]["mixInputs"]["lo_frequency"] = frequency
        self.mixers[f"mixer_drive{self}"][0]["lo_frequency"] = frequency

    def set_readout_lo_frequency(self, frequency):
        # TODO: Maybe this can be moved to ``Channel``
        self.elements[f"readout{self}"]["mixInputs"]["lo_frequency"] = frequency
        self.mixers[f"mixer_readout{self}"][0]["lo_frequency"] = frequency

    def set_drive_if_frequency(self, frequency):
        self.elements[f"drive{self}"]["intermediate_frequency"] = frequency
        self.mixers[f"mixer_drive{self}"][0]["intermediate_frequency"] = frequency

    def set_readout_if_frequency(self, frequency):
        self.elements[f"readout{self}"]["intermediate_frequency"] = frequency
        self.mixers[f"mixer_readout{self}"][0]["intermediate_frequency"] = frequency

    def set_flux_frequency(self, frequency):
        self.elements[f"flux{self}"]["intermediate_frequency"] = frequency

    def register_drive_pulse(self, pulse):
        self.elements[f"drive{self}"]["operations"][pulse.serial] = pulse.serial

    def register_readout_pulse(self, pulse):
        self.elements[f"readout{self}"]["operations"][pulse.serial] = pulse.serial

    def register_flux_pulse(self, pulse):
        self.elements[f"flux{self}"]["operations"][pulse.serial] = pulse.serial


class QuantumMachinesPlatform(AbstractPlatform):
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

        settings = self.settings["settings"]
        self.simulation = settings["simulation"]
        self.simulation_duration = settings["simulation_duration"] // 4  # convert to clock cycles

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
        qm = instruments.pop("qm")
        # Register controllers in config and channels
        for controller, values in qm["controllers"].items():
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

        # Instantiate QuantumMachines manager
        if self.simulation:
            from qm.simulate.credentials import create_credentials

            self.manager = QuantumMachinesManager(qm["address"], qm["port"], credentials=create_credentials())
        else:
            self.manager = QuantumMachinesManager(qm["address"], qm["port"])

        self.reload_settings()

    def __repr__(self):
        return self.name

    def __getstate__(self):
        return {
            "name": self.name,
            "runcard": self.runcard,
            "settings": self.settings,
            "is_connected": self.is_connected,
        }

    def __setstate__(self, data):
        self.name = data.get("name")
        self.runcard = data.get("runcard")
        self.settings = data.get("settings")
        self.is_connected = data.get("is_connected")

    def _check_connected(self):
        if not self.is_connected:
            raise_error(RuntimeError, "Cannot access instrument because it is not connected.")

    def reload_settings(self):
        with open(self.runcard) as file:
            self.settings = yaml.safe_load(file)

        if self.is_connected:
            self.setup()

    def run_calibration(self, show_plots=False):  # pragma: no cover
        raise_error(NotImplementedError)

    def connect(self):
        """Connects to lab instruments using the details specified in the calibration settings."""
        if not self.is_connected:
            try:
                for name in self.instruments:
                    log.info(f"Connecting to {self.name} instrument {name}.")
                    self.instruments[name].connect()
                self.is_connected = True
            except Exception as exception:
                raise_error(
                    RuntimeError,
                    "Cannot establish connection to " f"{self.name} instruments. " f"Error captured: '{exception}'",
                )

    def setup(self):
        if self.is_connected:
            for name, lo in self.local_oscillators.items():
                lo.setup(**self.settings["instruments"][name]["settings"])
        else:
            log.warn("There is no connection to local oscillators. Frequencies were not set.")
            for name, lo in self.local_oscillators.items():
                inst = self.settings["instruments"][name]
                frequency = inst["settings"]["frequency"]
                for channel_name in inst["channels"]:
                    Channel(channel_name).set_lo_frequency(frequency)

    def start(self):
        if self.is_connected:
            for lo in self.local_oscillators.values():
                lo.start()

    def stop(self):
        if self.is_connected:
            for lo in self.local_oscillators.values():
                lo.stop()

    def disconnect(self):
        # TODO: Disconnect from self.manager
        if self.is_connected:
            for lo in self.local_oscillators.values():
                lo.disconnect()
            self.is_connected = False

    def register_waveform(self, pulse, mode="i"):
        # example waveforms
        # "zero_wf": {"type": "constant", "sample": 0.0},
        # "x90_wf": {"type": "arbitrary", "samples": x90_wf.tolist()},
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
        rotation_angle = qubit.characterization.rotation_angle
        self.config["integration_weights"] = {
            f"rotated_cosine_weights_{qubit}": {
                "cosine": [(np.cos(rotation_angle), readout_len)],
                "sine": [(-np.sin(rotation_angle), readout_len)],
            },
            f"rotated_sine_weights_{qubit}": {
                "cosine": [(np.sin(rotation_angle), readout_len)],
                "sine": [(np.cos(rotation_angle), readout_len)],
            },
            f"rotated_minus_sine_weights_{qubit}": {
                "cosine": [(-np.sin(rotation_angle), readout_len)],
                "sine": [(-np.cos(rotation_angle), readout_len)],
            },
        }

    def register_pulse(self, pulse):
        pulses = self.config["pulses"]
        if pulse.serial not in pulses:

            qubit = self.qubits[pulse.qubit]
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
                return f"drive{qubit}"

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
                return f"flux{qubit}"

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
                        "rotated_cos": f"rotated_cosine_weights_{pulse.qubit}",
                        "rotated_sin": f"rotated_sine_weights_{pulse.qubit}",
                        "rotated_minus_sin": f"rotated_minus_sine_weights_{pulse.qubit}",
                    },
                    "digital_marker": "ON",
                }
                qubit.set_readout_if_frequency(pulse.frequency)
                qubit.register_readout_pulse(pulse)
                return f"readout{qubit}"

            else:
                raise_error(TypeError, f"Unknown pulse type {pulse.type.name}.")

    def execute_program(self, program):
        if self.simulation:
            # controller_connections = create_simulator_controller_connections(3)
            simulation_config = SimulationConfig(duration=self.simulation_duration)
            return self.manager.simulate(self.config, program, simulation_config)
        else:
            machine = self.manager.open_qm(self.config)
            return machine.execute(program)

    def execute_pulse_sequence(self, sequence, nshots=None):
        from qm.qua import play, program

        # register pulses in Quantum Machines config
        targets = [self.register_pulse(pulse) for pulse in sequence]
        # play pulses using QUA
        with program() as experiment:
            for pulse, target in zip(sequence, targets):
                play(pulse.serial, target)

        return self.execute_program(experiment)
