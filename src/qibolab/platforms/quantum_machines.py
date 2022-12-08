from types import SimpleNamespace

import yaml
from qibo.config import log, raise_error
from qm.QuantumMachinesManager import QuantumMachinesManager

from qibolab.instruments.rohde_schwarz import SGS100A


def IQ_imbalance(g, phi):
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


DEFAULT_CONFIG = {
    "version": 1,
    "controllers": {},
    "elements": {
        "qubit0": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "x90": "x90_pulse",
                "x180": "x180_pulse",
                "y90": "y90_pulse",
            },
        },
        "qubit1": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit1",
            },
            "intermediate_frequency": qubit1_IF,
            "operations": {
                "x90": "x90_pulse1",
                "x180": "x180_pulse1",
                "y90": "y90_pulse1",
            },
        },
        "resonator0": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator",
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": smearing,
        },
        "resonator1": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": resonator_LO,
                "mixer": "mixer_resonator1",
            },
            "intermediate_frequency": resonator1_IF,
            "operations": {
                "readout": "readout_pulse1",
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": smearing,
        },
        "flux_line0": {
            "singleInput": {
                "port": ("con1", 7),
            },
            "intermediate_frequency": flux_line_IF,
            "operations": {
                "cw": "const_flux_pulse",
            },
        },
        "flux_line1": {
            "singleInput": {
                "port": ("con1", 8),
            },
            "intermediate_frequency": flux_line_IF,
            "operations": {
                "cw": "const_flux_pulse1",
            },
        },
    },
    "pulses": {
        "const_flux_pulse": {
            "operation": "control",
            "length": const_flux_len,
            "waveforms": {
                "single": "const_flux_wf",
            },
        },
        "const_flux_pulse1": {
            "operation": "control",
            "length": const_flux_len1,
            "waveforms": {
                "single": "const_flux_wf1",
            },
        },
        "x90_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "x90_wf",
                "Q": "x90_der_wf",
            },
        },
        "x180_pulse": {
            "operation": "control",
            "length": x180_len,
            "waveforms": {
                "I": "x180_wf",
                "Q": "x180_der_wf",
            },
        },
        "y90_pulse": {
            "operation": "control",
            "length": y90_len,
            "waveforms": {
                "I": "y90_der_wf",
                "Q": "y90_wf",
            },
        },
        "x90_pulse1": {
            "operation": "control",
            "length": x90_len1,
            "waveforms": {
                "I": "x90_wf1",
                "Q": "x90_der_wf1",
            },
        },
        "x180_pulse1": {
            "operation": "control",
            "length": x180_len1,
            "waveforms": {
                "I": "x180_wf1",
                "Q": "x180_der_wf1",
            },
        },
        "y90_pulse1": {
            "operation": "control",
            "length": y90_len1,
            "waveforms": {
                "I": "y90_der_wf1",
                "Q": "y90_wf1",
            },
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "I": "readout_wf",
                "Q": "zero_wf",
            },
            "integration_weights": {
                "rotated_cos": "rotated_cosine_weights",
                "rotated_sin": "rotated_sine_weights",
                "rotated_minus_sin": "rotated_minus_sine_weights",
            },
            "digital_marker": "ON",
        },
        "readout_pulse1": {
            "operation": "measurement",
            "length": readout_len1,
            "waveforms": {
                "I": "readout_wf1",
                "Q": "zero_wf",
            },
            "integration_weights": {
                "rotated_cos": "rotated_cosine_weights1",
                "rotated_sin": "rotated_sine_weights1",
                "rotated_minus_sin": "rotated_minus_sine_weights1",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "x90_wf": {"type": "arbitrary", "samples": x90_wf.tolist()},
        "x90_der_wf": {"type": "arbitrary", "samples": x90_der_wf.tolist()},
        "x180_wf": {"type": "arbitrary", "samples": x180_wf.tolist()},
        "x180_der_wf": {"type": "arbitrary", "samples": x180_der_wf.tolist()},
        "y90_wf": {"type": "arbitrary", "samples": y90_wf.tolist()},
        "y90_der_wf": {"type": "arbitrary", "samples": y90_der_wf.tolist()},
        "x90_wf1": {"type": "arbitrary", "samples": x90_wf1.tolist()},
        "x90_der_wf1": {"type": "arbitrary", "samples": x90_der_wf1.tolist()},
        "x180_wf1": {"type": "arbitrary", "samples": x180_wf1.tolist()},
        "x180_der_wf1": {"type": "arbitrary", "samples": x180_der_wf1.tolist()},
        "y90_wf1": {"type": "arbitrary", "samples": y90_wf1.tolist()},
        "y90_der_wf1": {"type": "arbitrary", "samples": y90_der_wf1.tolist()},
        "readout_wf": {"type": "constant", "sample": readout_amp},
        "readout_wf1": {"type": "constant", "sample": readout_amp1},
        "const_flux_wf": {"type": "constant", "sample": const_flux_amp},
        "const_flux_wf1": {"type": "constant", "sample": const_flux_amp1},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "rotated_cosine_weights": {
            "cosine": [(np.cos(rotation_angle), readout_len)],
            "sine": [(-np.sin(rotation_angle), readout_len)],
        },
        "rotated_sine_weights": {
            "cosine": [(np.sin(rotation_angle), readout_len)],
            "sine": [(np.cos(rotation_angle), readout_len)],
        },
        "rotated_minus_sine_weights": {
            "cosine": [(-np.sin(rotation_angle), readout_len)],
            "sine": [(-np.cos(rotation_angle), readout_len)],
        },
        "rotated_cosine_weights1": {
            "cosine": [(np.cos(rotation_angle1), readout_len1)],
            "sine": [(-np.sin(rotation_angle1), readout_len1)],
        },
        "rotated_sine_weights1": {
            "cosine": [(np.sin(rotation_angle1), readout_len1)],
            "sine": [(np.cos(rotation_angle1), readout_len1)],
        },
        "rotated_minus_sine_weights1": {
            "cosine": [(-np.sin(rotation_angle1), readout_len1)],
            "sine": [(-np.cos(rotation_angle1), readout_len1)],
        },
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance(mixer_qubit_g, mixer_qubit_phi),
            }
        ],
        "mixer_qubit1": [
            {
                "intermediate_frequency": qubit1_IF,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance(mixer_qubit_g, mixer_qubit_phi),
            }
        ],
        "mixer_resonator": [
            {
                "intermediate_frequency": resonator_IF,
                "lo_frequency": resonator_LO,
                "correction": IQ_imbalance(mixer_resonator_g, mixer_resonator_phi),
            }
        ],
        "mixer_resonator1": [
            {
                "intermediate_frequency": resonator1_IF,
                "lo_frequency": resonator_LO,
                "correction": IQ_imbalance(mixer_resonator_g, mixer_resonator_phi),
            }
        ],
    },
}


class Qubit:
    def __init__(self, number, readout, drive, flux):
        self.number = number
        self.readout = readout
        self.drive = drive
        self.flux = flux

    def __repr__(self):
        return f"<Qubit {self.number}>"


class Channel:

    instances = {}

    def __new__(cls, name, role):
        if name not in cls.instances:
            print(name, role)
            cls.instances[name] = super().__new__(cls)
        elif role != cls.instances[name].role:
            raise_error(
                ValueError, f"Channel {name} already exists for {existing_role}. " f"Cannot recreate for {role}"
            )
        return cls.instances[name]

    def set_ports(cls, name, ports):
        channel = cls.instances[name]
        if channel.role == "flux":
            self.flux_port = ports[0]
        else:
            self.i_port, self.q_port = ports

    def set_local_oscillator(cls, name, local_oscillator):
        cls.instances[name].local_oscillator = local_oscillator

    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.i_port = None
        self.q_port = None
        self.flux_port = None
        self.local_oscillator = None

    def __repr__(self):
        return f"<Channel {self.name} | {self.role}>"


class QuantumMachinesPlatform:
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
        # self.channels = self.settings["channels"]
        # self.qubit_channel_map = self.settings["qubit_channel_map"]
        # create channels and qubits
        self.qubits = []
        ch_roles = ["readout", "drive", "flux"]
        for q, ch_names in self.settings["qubit_channel_map"].items():
            channels = [Channel(name, role) for name, role in zip(ch_names, ch_roles)]
            self.qubits.append(Qubit(q, *channels))

        instruments = self.settings["instruments"]
        # Instantiate Quantum Machines manager
        qm = instruments.pop("qm")
        self.manager = QuantumMachinesManager(qmm["address"])
        # Register controllers in config
        for name, values in qm["controllers"].items():
            self.config["controllers"][controller] = values["ports"]
            for channel, ports in values["channel_port_map"].items():
                Channel.set_ports(channel, ports)

        # Instantiate local oscillators
        self.local_oscillators = {}
        for name, value in instruments.items():
            lo = SGS100A(name, value["address"])
            lo.setup(**value["settings"])
            self.local_oscillators[name] = lo
            Channel.set_local_oscillator(value["channel"])

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

        self.hardware_avg = self.settings["settings"]["hardware_avg"]
        self.sampling_rate = self.settings["settings"]["sampling_rate"]
        self.repetition_duration = self.settings["settings"]["repetition_duration"]

        # Load Characterization settings
        self.characterization = self.settings["characterization"]
        # Load Native Gates
        self.native_gates = self.settings["native_gates"]

        if self.is_connected:
            self.setup()

    def run_calibration(self, show_plots=False):  # pragma: no cover
        raise NotImplementedError

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
        if not self.is_connected:
            raise_error(
                RuntimeError,
                "There is no connection to the instruments, the setup cannot be completed",
            )
        for name, lo in self.local_oscillators.items():
            lo.setup(**self.settings["instruments"][name]["settings"])

    def start(self):
        if self.is_connected:
            for lo in self.local_oscillators.values():
                lo.start()

    def stop(self):
        if self.is_connected:
            for lo in self.local_oscillators.values():
                lo.stop()

    def disconnect(self):
        # TODO: Disconnect from self.qmm
        if self.is_connected:
            for lo in self.local_oscillators.values():
                lo.disconnect()
            self.is_connected = False

    def execute_pulse_sequence(self, sequence, nshots=None):  # pragma: no cover
        raise NotImplementedError
