import pathlib

from qibolab.components import AcquisitionChannel, Channel, DcChannel, IqChannel
from qibolab.identifier import ChannelId
from qibolab.instruments.qm import Octave, QmConfigs, QmController
from qibolab.parameters import ConfigKinds
from qibolab.platform import Platform
from qibolab.qubits import Qubit

FOLDER = pathlib.Path(__file__).parent

# Register QM-specific configurations for parameters loading
ConfigKinds.extend([QmConfigs])


def create():
    """Example platform using Quantum Machines instruments."""
    qubits = {i: Qubit.default(i) for i in range(5)}

    # Create channels and connect to instrument ports
    # Readout
    channels: dict[ChannelId, Channel] = {}
    for q in qubits.values():
        channels[q.probe] = IqChannel(
            device="octave5", path="1", mixer=None, lo="probe_lo"
        )

    # Acquire
    for q in qubits.values():
        channels[q.acquisition] = AcquisitionChannel(
            device="octave5", path="1", twpa_pump=None, probe=q.probe
        )

    # Drive
    def define_drive(q: str, device: str, port: int, lo: str):
        drive = qubits[q].drive
        channels[drive] = IqChannel(device=device, path=str(port), mixer=None, lo=lo)

    define_drive(0, "octave5", 2, "0/drive_lo")
    define_drive(1, "octave5", 4, "12/drive_lo")
    define_drive(2, "octave5", 5, "12/drive_lo")
    define_drive(3, "octave6", 5, "3/drive_lo")
    define_drive(4, "octave6", 3, "4/drive_lo")

    # Flux
    for q, qubit in qubits.items():
        channels[qubit.flux] = DcChannel(device="con9", path=str(q + 3))

    octaves = {
        "octave5": Octave("octave5", port=11104, connectivity="con6"),
        "octave6": Octave("octave6", port=11105, connectivity="con8"),
    }
    controller = QmController(
        address="0.0.0.0:0",
        octaves=octaves,
        channels=channels,
        calibration_path=FOLDER,
        script_file_name="qua_script.py",
    )
    instruments = {"qm": controller}
    return Platform.load(path=FOLDER, instruments=instruments, qubits=qubits)
