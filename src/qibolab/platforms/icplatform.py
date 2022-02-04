import pathlib
import yaml
from qibo.config import raise_error, log

class ICPlatform:
    """Platform for controlling quantum devices.

    The path of the calibration json can be provided using the
    ``"CALIBRATION_PATH"`` environment variable.
    If no path is provided the default path (``platforms/tiiq_settings.json``)
    will be used.

    Args:
        name (str): name of the platform stored in a yaml file in qibolab/platforms.
    """

    def __init__(self, name):
        log.info(f"Loading platform {name}.")
        self.name = name
        # Load calibration settings
        self.calibration_path = pathlib.Path(__file__).parent / f"{name}.yml"
        with open(self.calibration_path, "r") as file:
            self._settings = yaml.safe_load(file)

        # Define references to instruments
        self.is_connected = False
        self._ic = None
        # instruments are connected in :meth:`qibolab.platform.Platform.start`

    def _check_connected(self):
        if not self.is_connected:
            raise_error(RuntimeError, "Cannot access instrument because it is not connected.")

    def connect(self):
        """Connects to lab instruments using the details specified in the calibration settings."""
        if not self.is_connected:
            log.info(f"Connecting to {self.name} instruments.")
            try:
                from qibolab.instruments.controller import InstrumentController
                self.ic = InstrumentController()
                instruments = self._settings.get("instruments")
                for name, params in instruments.items():
                    inst_type = params.get("type")
                    address = params.get("address")
                    self.ic.add_instrument(inst_type, name, address)
                self.is_connected = True
            except Exception as exception:
                raise_error(RuntimeError, "Cannot establish connection to "
                                         f"{self.name} instruments. "
                                         f"Error captured: '{exception}'")
        self.setup()

    def setup(self):
        """Configures instruments using the loaded calibration settings."""
        if self.is_connected:
            instruments = self._settings.get("instruments")
            for name, params in instruments.items():
                setup_params = params.get("setup_parameters")
                self.ic.setup_instrument(name, setup_params)
    
    def start(self):
        """Turns on the local oscillators.
        
        At this point, the pulse sequence have not been uploaded to the DACs, so they will not be started yet.
        """
        self.connect()
        self.ic.start_playback()

    def stop(self):
        """Turns off all the lab instruments."""
        self.ic.stop()

    def disconnect(self):
        """Disconnects from the lab instruments."""
        if self.is_connected:
            self.ic.close()

    def execute(self, sequence, nshots=None):
        """Executes a pulse sequence.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Pulse sequence to execute.
            nshots (int): Number of shots to sample from the experiment.
                If ``None`` the default value provided as hardware_avg in the
                calibration json will be used.

        Returns:
            Readout results acquired by :class:`qibolab.instruments.qblox.PulsarQRM`
            after execution.
        """
        if not self.is_connected:
            raise_error(RuntimeError, "Execution failed because instruments are not connected.")
        if nshots is None:
            nshots = self.hardware_avg
        
        sobj = {}
        robj = {}
        from qibolab.pulses import ReadoutPulse
        for pulse in sequence.pulses:
            # Assign pulses to each respective waveform generator
            if pulse.device not in sobj.keys():
                sobj[pulse.device] = []
            sobj[pulse.device].append(pulse)

            # Track each readout pulse and frequency to its readout device
            if isinstance(pulse, ReadoutPulse):
                if pulse.adc not in robj.keys():
                    robj[pulse.adc] = []
                    robj[pulse.adc].append(pulse.frequency)

        # Translate and upload the pulse for each device
        for device, seq in sobj.items():
            self.ic.translate_and_upload(device, seq, nshots)

        # Trigger the experiment
        self.ic.trigger_experiment()

        # Fetch the experiment results
        result = self.ic.result(robj)
        return result
