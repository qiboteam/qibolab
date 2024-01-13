"""Qblox Cluster QCM driver."""

from qibolab.instruments.abstract import Instrument
from qibolab.instruments.qblox.port import QbloxInputPort, QbloxOutputPort
from qibolab.pulses import Pulse, PulseSequence
from qibolab.qubits import Qubit


class ClusterModule(Instrument):
    """This class defines common features shared by all Qblox modules (QCM- BB,
    QCM-RF, QRM-RF).

    It serves as a foundational class, unifying the behavior of the
    three distinct modules. All module-specific classes are intended to
    inherit from this base class.
    """

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
        self._ports: dict = {}

    def ports(self, name: str, out: bool = True):
        """Adds an entry to the dictionary `self._ports` with key 'name' and
        value a `QbloxOutputPort` (or `QbloxInputPort` if `out=False`) object.
        To the object is assigned the provided name, and the `port_number` is
        automatically determined based on the number of ports of the same type
        inside `self._ports`.

        Returns this port object.

        Example:
        >>> qrm_module = ClusterQRM_RF("qrm_rf", f"{IP_ADDRESS}:{SLOT_IDX}")
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

    def start(self):
        """Empty method to comply with Instrument interface."""
        pass

    def disconnect(self):
        """Empty method to comply with Instrument interface."""
        self.is_connected = False
        self.device = None
