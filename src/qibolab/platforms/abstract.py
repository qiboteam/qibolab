from qibo.config import log
from abc import ABC, abstractmethod
import yaml
from qibo.core import circuit
from qibolab.circuit import PulseSequence
from qibolab.pulses import Pulse, ReadoutPulse, Rectangular, Gaussian, Drag

class AbstractPlatform(ABC):
    """Abstract platform for controlling quantum devices.

    Args:
        name (str): name of the platform.
        runcard (str): path to the yaml file containing the platform setup.
    """
    def __init__(self, name, runcard):
        log.info(f"Loading platform {name} from runcard {runcard}")
        self.name = name
        self.runcard = runcard
        self.is_connected = False        
        # Load platform settings
        with open(runcard, "r") as file:
            self.settings = yaml.safe_load(file)
            
        self.instruments = {}
        # Instantiate instruments 
        for name in self.settings['instruments']:
            lib = self.settings['instruments'][name]['lib']
            i_class = self.settings['instruments'][name]['class']
            address = self.settings['instruments'][name]['address']
            from importlib import import_module
            InstrumentClass = getattr(import_module(f"qibolab.instruments.{lib}"), i_class)
            instance = InstrumentClass(name, address)
            self.instruments[name] = instance    

    def __getstate__(self):
        return {
            "name": self.name,
            "runcard": self.runcard,
            "settings": self.settings,
            "is_connected": self.is_connected
        }

    def __setstate__(self, data):
        self.name = data.get("name")
        self.runcard = data.get("runcard")
        self.settings = data.get("settings")
        self.is_connected = data.get("is_connected")

    def _check_connected(self):
        if not self.is_connected:
            from qibo.config import raise_error
            raise_error(RuntimeError, "Cannot access instrument because it is not connected.")

    def reload_settings(self):
        with open(self.runcard, "r") as file:
            self.settings = yaml.safe_load(file)
        self.setup()

    @abstractmethod
    def run_calibration(self, show_plots=False):  # pragma: no cover
        """Executes calibration routines and updates the settings yml file"""
        from qibolab.calibration import calibration
        ac = calibration.Calibration(self)
        ac.auto_calibrate_plaform()     
        # update instruments with new calibration settings
        self.reload_settings()

    def connect(self):
        """Connects to lab instruments using the details specified in the calibration settings."""
        if not self.is_connected:
            try:
                for name in self.instruments:
                    log.info(f"Connecting to {self.name} instrument {name}.")
                    self.instruments[name].connect()
                self.is_connected = True
            except Exception as exception:
                from qibo.config import raise_error
                raise_error(RuntimeError, "Cannot establish connection to "
                            f"{self.name} instruments. "
                            f"Error captured: '{exception}'")

    def setup(self):
        self.hardware_avg = self.settings['settings']['hardware_avg']
        self.sampling_rate = self.settings['settings']['sampling_rate']
        self.repetition_duration = self.settings['settings']['repetition_duration']
        self.minimum_delay_between_instructions = self.settings['settings']['minimum_delay_between_instructions']

        self.qubits = self.settings['qubits']
        self.topology = self.settings['topology']
        self.channels = self.settings['channels']
        self.qubit_channel_map = self.settings['qubit_channel_map']
        
        # Generate qubit_instrument_map from qubit_channel_map and the instruments' channel_port_maps
        self.qubit_instrument_map = {}
        for qubit in self.qubit_channel_map:
            self.qubit_instrument_map[qubit] = [None, None, None]
            for name in self.instruments:
                if 'channel_port_map' in self.settings['instruments'][name]['settings']:
                    for channel in self.settings['instruments'][name]['settings']['channel_port_map']:
                        if channel in self.qubit_channel_map[qubit]:
                             self.qubit_instrument_map[qubit][self.qubit_channel_map[qubit].index(channel)] = name
        # Generate ro_channel[qubit], qd_channel[qubit], qf_channel[qubit], qrm[qubit], qcm[qubit], lo_qrm[qubit], lo_qcm[qubit]
        self.ro_channel = {}
        self.qd_channel = {}
        self.qf_channel = {}
        self.qrm = {}
        self.qcm = {}
        for qubit in self.qubit_channel_map:
            self.ro_channel[qubit] = self.qubit_channel_map[qubit][0]
            self.qd_channel[qubit] = self.qubit_channel_map[qubit][1]
            self.qf_channel[qubit] = self.qubit_channel_map[qubit][2]

            if not self.qubit_instrument_map[qubit][0] is None:
                self.qrm[qubit]  = self.instruments[self.qubit_instrument_map[qubit][0]]
            if not self.qubit_instrument_map[qubit][1] is None:
                self.qcm[qubit]  = self.instruments[self.qubit_instrument_map[qubit][1]]
            # TODO: implement qf modules


        # Load Native Gates
        self.native_gates = self.settings['native_gates']

        if self.is_connected:
            for name in self.instruments:
                # Set up every with the platform settings and the instrument settings 
                self.instruments[name].setup(**self.settings['settings'], **self.settings['instruments'][name]['settings'])
        
        # Load Characterization settings
        self.characterization = self.settings['characterization']

    def start(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].start()

    def stop(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].stop()

    def disconnect(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].disconnect()
            self.is_connected = False

    def __call__(self, sequence, nshots=None):
        return self.execute_pulse_sequence(sequence, nshots)

    @abstractmethod
    def execute_pulse_sequence(self, sequence, nshots=None):  # pragma: no cover
        """Executes a pulse sequence.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration yml will be used.

        Returns:
            Readout results acquired by after execution.
        """
        raise NotImplementedError


    def execute_circuit(self, circuit: circuit.Circuit, nshots=None):
        """Executes a quantum circuit.

        Args:
            circuit (:class:`qibo.core.circuit.Circuit`): Circuit to execute.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration yml will be used.

        Returns:
            Readout results acquired by after execution.
        """

        sequence = PulseSequence()
        sequence.platform = self
        for gate in circuit.queue:
            # Cast qibo gate into qibolab hardware gate
            gate.__class__.replace('qibo.core.gates', 'qibolab.gates')
            gate.to_sequence(sequence)
        circuit.measurement_gate.to_sequence(sequence)


    def RX90_pulse(self, qubit, start = 0, phase = 0):
        qd_duration = self.settings['native_gates']['single_qubit'][qubit]['RX']['duration'] 
        qd_frequency = self.settings['native_gates']['single_qubit'][qubit]['RX']['frequency']
        qd_amplitude = self.settings['native_gates']['single_qubit'][qubit]['RX']['amplitude'] / 2
        qd_shape = self.settings['native_gates']['single_qubit'][qubit]['RX']['shape']
        qd_channel = self.settings['qubit_channel_map'][qubit][1]
        from qibolab.pulses import Pulse
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, phase, qd_shape, qd_channel)

    
    def RX_pulse(self, qubit, start = 0, phase = 0):
        qd_duration = self.settings['native_gates']['single_qubit'][qubit]['RX']['duration'] 
        qd_frequency = self.settings['native_gates']['single_qubit'][qubit]['RX']['frequency']
        qd_amplitude = self.settings['native_gates']['single_qubit'][qubit]['RX']['amplitude']
        qd_shape = self.settings['native_gates']['single_qubit'][qubit]['RX']['shape']
        qd_channel = self.settings['qubit_channel_map'][qubit][1]
        from qibolab.pulses import Pulse
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, phase, qd_shape, qd_channel)

    def MZ_pulse(self, qubit, start, phase = 0):
        ro_duration = self.settings['native_gates']['single_qubit'][qubit]['MZ']['duration'] 
        ro_frequency = self.settings['native_gates']['single_qubit'][qubit]['MZ']['frequency']
        ro_amplitude = self.settings['native_gates']['single_qubit'][qubit]['MZ']['amplitude']
        ro_shape = self.settings['native_gates']['single_qubit'][qubit]['MZ']['shape']     
        ro_channel = self.settings['qubit_channel_map'][qubit][0]
        from qibolab.pulses import ReadoutPulse
        return ReadoutPulse(start, ro_duration, ro_amplitude, ro_frequency, phase, ro_shape, ro_channel)
    
    
    def qubit_drive_pulse(self, qubit, start, duration, phase = 0):
        qd_frequency = self.settings['native_gates']['single_qubit'][qubit]['RX']['frequency']
        qd_amplitude = self.settings['native_gates']['single_qubit'][qubit]['RX']['amplitude']
        qd_shape = self.settings['native_gates']['single_qubit'][qubit]['RX']['shape']
        qd_channel = self.settings['qubit_channel_map'][qubit][1]
        from qibolab.pulses import Pulse
        return Pulse(start, duration, qd_amplitude, qd_frequency, phase, qd_shape, qd_channel)


    def qubit_readout_pulse(self, qubit, start, phase = 0):
        ro_duration = self.settings['native_gates']['single_qubit'][qubit]['MZ']['duration'] 
        ro_frequency = self.settings['native_gates']['single_qubit'][qubit]['MZ']['frequency']
        ro_amplitude = self.settings['native_gates']['single_qubit'][qubit]['MZ']['amplitude']
        ro_shape = self.settings['native_gates']['single_qubit'][qubit]['MZ']['shape']     
        ro_channel = self.settings['qubit_channel_map'][qubit][0]
        from qibolab.pulses import ReadoutPulse
        return ReadoutPulse(start, ro_duration, ro_amplitude, ro_frequency, phase, ro_shape, ro_channel)