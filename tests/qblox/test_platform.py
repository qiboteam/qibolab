from qibolab._core.components.channels import IqChannel
from qibolab._core.instruments.qblox.platform import build_channels, map_ports
from qibolab._core.qubits import Qubit

CLUSTER = {"qcm_rf0": (2, {1: [0]}), "qrm_rf0": (4, {"io1": [0]})}


def _qubits():
    return {0: Qubit.default(0, drive_extra={(1, 2): "0/drive12", 1: "0/drive1"})}


def test_map_ports_drive_extra():
    channels, _ = map_ports(CLUSTER, _qubits())
    drive = channels["0/drive"]
    for extra in ["0/drive12", "0/drive1"]:
        assert isinstance(channels[extra], IqChannel)
        assert channels[extra].path == drive.path == "2/o1"


def test_build_channels_drive_extra():
    channels, _ = build_channels(CLUSTER, _qubits())
    drive = channels["0/drive"]
    for extra in ["0/drive12", "0/drive1"]:
        assert channels[extra].path == drive.path
        assert channels[extra].lo == drive.lo == "qcm_rf0/o1/lo"
        assert channels[extra].mixer == drive.mixer == "qcm_rf0/o1/mixer"
