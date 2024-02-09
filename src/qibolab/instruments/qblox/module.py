"""Qblox Cluster QCM driver."""

from abc import abstractmethod

from qblox_instruments.qcodes_drivers.qcm_qrm import QcmQrm as QbloxQrmQcm
from qibo.config import log

from qibolab.instruments.abstract import Instrument
from qibolab.pulses import Pulse, PulseSequence
from qibolab.qubits import Qubit

from .port import QbloxInputPort, QbloxOutputPort
from .sequencer import Sequencer


class ClusterModule(Instrument):
    """This class defines common features shared by all Qblox modules (QCM-BB,
    QCM-RF, QRM-RF).

    It serves as a foundational class, unifying the behavior of the
    three distinct modules. All module-specific classes are intended to
    inherit from this base class.
    """

    FREQUENCY_LIMIT = 500e6  # 500Mhz
    DEFAULT_SEQUENCERS_VALUES = {
        "cont_mode_en_awg_path0": False,
        "cont_mode_en_awg_path1": False,
        "cont_mode_waveform_idx_awg_path0": 0,
        "cont_mode_waveform_idx_awg_path1": 0,
        "marker_ovr_en": True,  # Default after reboot = False
        "marker_ovr_value": 15,  # Default after reboot = 0
        "mixer_corr_gain_ratio": 1,
        "mixer_corr_phase_offset_degree": 0,
        "offset_awg_path0": 0,
        "offset_awg_path1": 0,
        "sync_en": False,  # Default after reboot = False
        "upsample_rate_awg_path0": 0,
        "upsample_rate_awg_path1": 0,
    }

    def __init__(self, name: str, address: str):
        super().__init__(name, address)
        self.device: QbloxQrmQcm = None
        self.channel_map: dict = {}
        self._ports: dict = {}
        self._device_num_sequencers: int
        # TODO: we can create only list and put three flags: free, used, unused
        self._free_sequencers_numbers: list[int] = []
        self._used_sequencers_numbers: list[int] = []
        self._unused_sequencers_numbers: list[int] = []
        self._debug_folder: str = ""
        self._sequencers: dict[str, list[Sequencer]] = {}

    def ports(self, name: str, out: bool = True):
        """Adds an entry to the dictionary `self._ports` with key 'name' and
        value a `QbloxOutputPort` (or `QbloxInputPort` if `out=False`) object.
        To the object is assigned the provided name, and the `port_number` is
        automatically determined based on the number of ports of the same type
        inside `self._ports`.

        Returns this port object.

        Example:
        >>> qrm_module = QrmRf("qrm_rf", f"{IP_ADDRESS}:{SLOT_IDX}")
        >>> output_port = qrm_module.add_port("o1")
        >>> input_port = qrm_module.add_port("i1", out=False)
        >>> qrm_module.ports
        {
            'o1': QbloxOutputPort(module=qrm_module, port_number=0, port_name='o1'),
            'i1': QbloxInputPort(module=qrm_module, port_number=0, port_name='i1')
        }
        """

        def count(cls):
            return len(list(filter(lambda x: isinstance(x, cls), self._ports.values())))

        port_cls = QbloxOutputPort if out else QbloxInputPort
        self._ports[name] = port_cls(self, port_number=count(port_cls), port_name=name)
        return self._ports[name]

    @abstractmethod
    def _setup_ports(self):
        pass

    def connect(self):
        """Connects to the instrument using the instrument settings in the
        runcard.

        Once connected, it creates port classes with properties mapped
        to various instrument parameters, and initialises the the
        underlying device parameters. It uploads to the module the port
        settings loaded from the runcard.
        """
        if self.is_connected:
            return
        # test connection with module. self.device is initialized in QbloxController connect()
        if not self.device.present():
            raise ConnectionError(f"Module {self.device.name} not present")
        # once connected, initialise the parameters of the device to the default values
        self._device_num_sequencers = len(self.device.sequencers)
        self._set_default_values()
        # then set the value loaded from the runcard
        try:
            self._setup_ports()
        except Exception as error:
            raise RuntimeError(
                f"Unable to initialize port parameters on module {self.name}: {error}"
            )
        self.is_connected = True

    def clone_sequencer_params(self, first_sequencer: int, next_sequencer: int):
        """Clone the values of all writable parameters from the first_sequencer
        into the next_sequencer."""
        for parameter in self.device.sequencers[first_sequencer].parameters:
            # exclude read-only parameter `sequence` and others that have wrong default values (qblox bug)
            if not parameter in [
                "sequence",
                "thresholded_acq_marker_address",
                "thresholded_acq_trigger_address",
            ]:
                value = self.device.sequencers[first_sequencer].get(
                    param_name=parameter
                )
                if value:
                    target = self.device.sequencers[next_sequencer]
                    target.set(parameter, value)

    def filter_port_pulse(
        self, pulses: PulseSequence, qubits: dict, port_obj: QbloxOutputPort
    ) -> PulseSequence:
        """Filters the pulses and return a new pulse sequence containing only
        pulses relative to the specified port (port_obj).

        Additionally builds the channel_map attribute which maps the
        channel name to the channel object.
        """
        for qubit in qubits.values():
            qubit: Qubit
            for channel in qubit.channels:
                if channel.port == port_obj:
                    self.channel_map[channel.name] = channel
                    return pulses.get_channel_pulses(channel.name)

    def get_if(self, pulse: Pulse):
        """Returns the intermediate frequency needed to synthesise a pulse
        based on the port lo frequency.

        Note:
        - ClusterQCM_BB has no external neither internal local oscillator so its
        _lo should be always zero.
        """

        _rf = pulse.frequency
        _lo = self.channel_map[pulse.channel].lo_frequency
        _if = _rf - _lo
        if abs(_if) > self.FREQUENCY_LIMIT:
            raise Exception(
                f"""
            Pulse frequency {_rf:_} cannot be synthesised with current lo frequency {_lo:_}.
            The intermediate frequency {_if:_} would exceed the maximum frequency of {self.FREQUENCY_LIMIT:_}
            """
            )
        return _if

    def play_sequence(self):
        """Plays the sequence of pulses.

        Starts the sequencers needed to play the sequence of pulses.
        """
        # Start used sequencers
        for sequencer_number in self._used_sequencers_numbers:
            self.device.start_sequencer(sequencer_number)

    def disconnect(self):
        """Stops all sequencers, disconnect all the outputs from the AWG paths
        of the sequencers.

        If the module is a QRM-RF disconnect all the inputs from the
        acquisition paths of the sequencers.
        """
        for sequencer_number in self._used_sequencers_numbers:
            state = self.device.get_sequencer_state(sequencer_number)
            if state.status != "STOPPED":
                log.warning(
                    f"Device {self.device.sequencers[sequencer_number].name} did not stop normally\nstate: {state}"
                )
        self.device.stop_sequencer()
        self.device.disconnect_outputs()
        if self.device.is_qrm_type():
            self.device.disconnect_inputs()
        self.is_connected = False
        self.device = None
