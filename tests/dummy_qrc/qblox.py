import pathlib

from qibolab.channels import Channel
from qibolab.instruments.qblox.cluster import (
    Cluster,
    Cluster_Settings,
    ReferenceClockSource,
)
from qibolab.instruments.qblox.cluster_qcm_bb import (
    ClusterQCM_BB,
    ClusterQCM_BB_Settings,
)
from qibolab.instruments.qblox.cluster_qcm_rf import (
    ClusterQCM_RF,
    ClusterQCM_RF_Settings,
)
from qibolab.instruments.qblox.cluster_qrm_rf import (
    ClusterQRM_RF,
    ClusterQRM_RF_Settings,
)
from qibolab.instruments.qblox.controller import QbloxController
from qibolab.instruments.qblox.port import (
    ClusterBB_OutputPort_Settings,
    ClusterRF_OutputPort_Settings,
    QbloxInputPort_Settings,
)
from qibolab.instruments.rohde_schwarz import SGS100A
from qibolab.platform import Platform
from qibolab.serialize import load_qubits, load_runcard, load_settings

NAME = "qblox"
ADDRESS = "192.168.0.6"
TIME_OF_FLIGHT = 500
RUNCARD = pathlib.Path(__file__).parent / "qblox.yml"

instruments_settings = {
    "cluster": Cluster_Settings(reference_clock_source=ReferenceClockSource.INTERNAL),
    "qrm_rf_a": ClusterQRM_RF_Settings(
        {
            "o1": ClusterRF_OutputPort_Settings(
                channel="L3-25_a",
                attenuation=38,
                lo_frequency=7_255_000_000,
                gain=0.6,
            ),
            "i1": QbloxInputPort_Settings(
                channel="L2-5_a",
                acquisition_hold_off=TIME_OF_FLIGHT,
                acquisition_duration=900,
            ),
        }
    ),
    "qrm_rf_b": ClusterQRM_RF_Settings(
        {
            "o1": ClusterRF_OutputPort_Settings(
                channel="L3-25_b",
                attenuation=32,
                lo_frequency=7_850_000_000,
                gain=0.6,
            ),
            "i1": QbloxInputPort_Settings(
                channel="L2-5_b",
                acquisition_hold_off=TIME_OF_FLIGHT,
                acquisition_duration=900,
            ),
        }
    ),
    "qcm_rf0": ClusterQCM_RF_Settings(
        {
            "o1": ClusterRF_OutputPort_Settings(
                channel="L3-15",
                attenuation=20,
                lo_frequency=5_250_304_836,
                gain=0.470,
            )
        }
    ),
    "qcm_rf1": ClusterQCM_RF_Settings(
        {
            "o1": ClusterRF_OutputPort_Settings(
                channel="L3-11",
                attenuation=20,
                lo_frequency=5_052_833_073,
                gain=0.570,
            ),
            "o2": ClusterRF_OutputPort_Settings(
                channel="L3-12",
                attenuation=20,
                lo_frequency=5_995_371_914,
                gain=0.655,
            ),
        }
    ),
    "qcm_rf2": ClusterQCM_RF_Settings(
        {
            "o1": ClusterRF_OutputPort_Settings(
                channel="L3-13",
                attenuation=20,
                lo_frequency=6_961_018_001,
                gain=0.550,
            ),
            "o2": ClusterRF_OutputPort_Settings(
                channel="L3-14",
                attenuation=20,
                lo_frequency=6_786_543_060,
                gain=0.596,
            ),
        }
    ),
    "qcm_bb0": ClusterQCM_BB_Settings(
        {"o1": ClusterBB_OutputPort_Settings(channel="L4-5", gain=0.5, offset=0.5507, qubit=0)}
    ),
    "qcm_bb1": ClusterQCM_BB_Settings(
        {
            "o1": ClusterBB_OutputPort_Settings(channel="L4-1", gain=0.5, offset=0.2227, qubit=1),
            "o2": ClusterBB_OutputPort_Settings(channel="L4-2", gain=0.5, offset=-0.3780, qubit=2),
            "o3": ClusterBB_OutputPort_Settings(channel="L4-3", gain=0.5, offset=-0.8899, qubit=3),
            "o4": ClusterBB_OutputPort_Settings(channel="L4-4", gain=0.5, offset=0.5890, qubit=4),
        }
    ),
    "twpa_pump": {"frequency": 6_535_900_000, "power": 4},
}


def create(runcard_path=RUNCARD):
    """QuantWare 5q-chip controlled using qblox cluster.

    Args:
        runcard (str): Path to the runcard file.
    """

    def instantiate_module(modules, cls, name, address, settings):
        module_settings = settings[name]
        modules[name] = cls(name=name, address=address, settings=module_settings)
        return modules[name]

    modules = {}

    cluster = Cluster(
        name="cluster",
        address="192.168.0.6",
        settings=instruments_settings["cluster"],
    )

    qrm_rf_a = instantiate_module(
        modules, ClusterQRM_RF, "qrm_rf_a", "192.168.0.6:10", instruments_settings
    )  # qubits q0, q1, q5
    qrm_rf_b = instantiate_module(
        modules, ClusterQRM_RF, "qrm_rf_b", "192.168.0.6:12", instruments_settings
    )  # qubits q2, q3, q4

    qcm_rf0 = instantiate_module(modules, ClusterQCM_RF, "qcm_rf0", "192.168.0.6:8", instruments_settings)  # qubit q0
    qcm_rf1 = instantiate_module(
        modules, ClusterQCM_RF, "qcm_rf1", "192.168.0.6:3", instruments_settings
    )  # qubits q1, q2
    qcm_rf2 = instantiate_module(
        modules, ClusterQCM_RF, "qcm_rf2", "192.168.0.6:4", instruments_settings
    )  # qubits q3, q4

    qcm_bb0 = instantiate_module(modules, ClusterQCM_BB, "qcm_bb0", "192.168.0.6:5", instruments_settings)  # qubit q0
    qcm_bb1 = instantiate_module(
        modules, ClusterQCM_BB, "qcm_bb1", "192.168.0.6:2", instruments_settings
    )  # qubits q1, q2, q3, q4

    # DEBUG: debug folder = report folder
    # import os
    # folder = os.path.dirname(runcard) + "/debug/"
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # for name in modules:
    #     modules[name]._debug_folder = folder

    controller = QbloxController("qblox_controller", cluster, modules)

    twpa_pump = SGS100A(name="twpa_pump", address="192.168.0.37")
    twpa_pump.frequency = instruments_settings["twpa_pump"]["frequency"]
    twpa_pump.power = instruments_settings["twpa_pump"]["power"]

    # Create channel objects
    channels = {}
    # readout
    channels["L3-25_a"] = Channel(name="L3-25_a", port=qrm_rf_a.ports["o1"])
    channels["L3-25_b"] = Channel(name="L3-25_b", port=qrm_rf_b.ports["o1"])

    # feedback
    channels["L2-5_a"] = Channel(name="L2-5_a", port=qrm_rf_a.ports["i1"])
    channels["L2-5_b"] = Channel(name="L2-5_b", port=qrm_rf_b.ports["i1"])

    # drive
    channels["L3-15"] = Channel(name="L3-15", port=qcm_rf0.ports["o1"])
    channels["L3-11"] = Channel(name="L3-11", port=qcm_rf1.ports["o1"])
    channels["L3-12"] = Channel(name="L3-12", port=qcm_rf1.ports["o2"])
    channels["L3-13"] = Channel(name="L3-13", port=qcm_rf2.ports["o1"])
    channels["L3-14"] = Channel(name="L3-14", port=qcm_rf2.ports["o2"])

    # flux
    channels["L4-5"] = Channel(name="L4-5", port=qcm_bb0.ports["o1"])
    channels["L4-1"] = Channel(name="L4-1", port=qcm_bb1.ports["o1"])
    channels["L4-2"] = Channel(name="L4-2", port=qcm_bb1.ports["o2"])
    channels["L4-3"] = Channel(name="L4-3", port=qcm_bb1.ports["o3"])
    channels["L4-4"] = Channel(name="L4-4", port=qcm_bb1.ports["o4"])

    # TWPA
    channels["L4-26"] = Channel(name="L4-4", port=None)

    # create qubit objects
    runcard = load_runcard(runcard_path)
    qubits, pairs = load_qubits(runcard)

    # assign channels to qubits
    for q in [0, 1]:
        qubits[q].readout = channels["L3-25_a"]
        qubits[q].feedback = channels["L2-5_a"]
        qubits[q].twpa = channels["L4-26"]
    for q in [2, 3, 4]:
        qubits[q].readout = channels["L3-25_b"]
        qubits[q].feedback = channels["L2-5_b"]
        qubits[q].twpa = channels["L4-26"]

    qubits[0].drive = channels["L3-15"]
    qubits[0].flux = channels["L4-5"]
    channels["L4-5"].qubit = qubits[0]
    for q in range(1, 5):
        qubits[q].drive = channels[f"L3-{10 + q}"]
        qubits[q].flux = channels[f"L4-{q}"]
        channels[f"L4-{q}"].qubit = qubits[q]

    # set maximum allowed bias
    for q in range(5):
        qubits[q].flux.max_bias = 2.5

    instruments = {controller.name: controller, twpa_pump.name: twpa_pump}
    settings = load_settings(runcard)
    return Platform("qblox", qubits, pairs, instruments, settings, resonator_type="2D")
