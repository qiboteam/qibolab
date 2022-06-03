from qibo.config import log
from abc import ABC, abstractmethod
import yaml
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
            # instance.__dict__.update(self.settings['settings'])
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
                from qibo.config import raise_error
                raise_error(RuntimeError, "Cannot establish connection to "
                            f"{self.name} instruments. "
                            f"Error captured: '{exception}'")

    def setup(self):
        self.__dict__.update(self.settings['settings'])
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
        self.lo_qrm = {}
        self.qcm = {}
        self.lo_qcm = {}
        for qubit in self.qubit_channel_map:
            self.ro_channel[qubit] = self.qubit_channel_map[qubit][0]
            self.qd_channel[qubit] = self.qubit_channel_map[qubit][1]
            self.qf_channel[qubit] = self.qubit_channel_map[qubit][2]

            if not self.qubit_instrument_map[qubit][0] is None:
                self.qrm[qubit]  = self.instruments[self.qubit_instrument_map[qubit][0]]
                self.lo_qrm[qubit] = self.instruments[self.settings['instruments'][self.qubit_instrument_map[qubit][0]]['settings']['lo']]
            if not self.qubit_instrument_map[qubit][1] is None:
                self.qcm[qubit]  = self.instruments[self.qubit_instrument_map[qubit][1]]
                self.lo_qcm[qubit] = self.instruments[self.settings['instruments'][self.qubit_instrument_map[qubit][1]]['settings']['lo']]
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
                calibration json will be used.

        Returns:
            Readout results acquired by after execution.
        """
        raise NotImplementedError


    def add_u3_to_pulse_sequence(self, pulse_sequence:PulseSequence, theta, phi, lam, qubit=1):
        """Convert a U3 gate into a sequence of pulses and add them to the PulseSequence object passed as a parameter.

        Args:
            pulse_sequence (PulseSequence): The PulseSequence object on which the new pulses will be added
            theta, phi, lam (float): Parameters of the U3 gate.
            qubit (int): the physical qubit to which the pulses are addressed
        """

        # Fetch pi/2 pulse from calibration
        RX = self.settings['native_gates']['single_qubit'][qubit]['RX']

        RX90_duration = RX['duration']
        RX90_amplitude = RX['amplitude']/2
        RX90_frequency = RX['frequency']
        RX90_shape = RX['shape']
        RX90_type = RX['type']
        RX90_channel = self.settings['qubit_channel_map'][qubit][1]

        # apply RZ(lam)
        pulse_sequence.phase += lam
        # apply RX(pi/2)
        pulse_sequence.add(Pulse(pulse_sequence.time, RX90_duration, RX90_amplitude, RX90_frequency, pulse_sequence.phase, RX90_shape, RX90_channel, RX90_type))
        pulse_sequence.time += RX90_duration
        # apply RZ(theta)
        pulse_sequence.phase += theta
        # apply RX(-pi/2)
        import math
        pulse_sequence.add(Pulse(pulse_sequence.time, RX90_duration, RX90_amplitude, RX90_frequency, pulse_sequence.phase - math.pi, RX90_shape, RX90_channel, RX90_type))
        pulse_sequence.time += RX90_duration
        # apply RZ(phi)
        pulse_sequence.phase += phi
        

    def add_measurement_to_pulse_sequence(self,  pulse_sequence:PulseSequence, qubit=1):
        """Add measurement gate to the PulseSequence object passed as a parameter.
        
        Args:
            pulse_sequence (PulseSequence): The PulseSequence object on which the new pulse will be added 
            qubit (int): the physical qubit to which the pulses are addressed
        """
        MZ = self.settings['native_gates']['single_qubit'][qubit]['MZ']
        MZ_duration = MZ['duration']
        MZ_channel = self.settings['qubit_channel_map'][qubit][0]
        pulse_sequence.add(ReadoutPulse(start = pulse_sequence.time, **MZ, phase = pulse_sequence.phase, channel = MZ_channel))
        pulse_sequence.time += MZ_duration


    def RX90_pulse(self, qubit, start, phase = 0):
        qd_duration = self.settings['native_gates']['single_qubit'][qubit]['RX']['duration'] 
        qd_frequency = self.settings['native_gates']['single_qubit'][qubit]['RX']['frequency']
        qd_amplitude = self.settings['native_gates']['single_qubit'][qubit]['RX']['amplitude'] / 2
        qd_shape = self.settings['native_gates']['single_qubit'][qubit]['RX']['shape']
        qd_channel = self.settings['qubit_channel_map'][qubit][1]
        from qibolab.pulses import Pulse
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, phase, qd_shape, qd_channel)

    
    def RX_pulse(self, qubit, start, phase = 0):
        qd_duration = self.settings['native_gates']['single_qubit'][qubit]['RX']['duration'] 
        qd_frequency = self.settings['native_gates']['single_qubit'][qubit]['RX']['frequency']
        qd_amplitude = self.settings['native_gates']['single_qubit'][qubit]['RX']['amplitude']
        qd_shape = self.settings['native_gates']['single_qubit'][qubit]['RX']['shape']
        qd_channel = self.settings['qubit_channel_map'][qubit][1]
        from qibolab.pulses import Pulse
        return Pulse(start, qd_duration, qd_amplitude, qd_frequency, phase, qd_shape, qd_channel)

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