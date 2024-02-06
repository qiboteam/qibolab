"""Qblox Cluster QCM driver."""

from qibolab.instruments.abstract import Instrument
from qibolab.instruments.qblox.port import QbloxInputPort, QbloxOutputPort


class ClusterModule(Instrument):
    """This class defines common features shared by all Qblox modules (QCM-BB,
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
