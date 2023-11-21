import pathlib

from qibolab.channels import Channel
from qibolab.instruments.qblox.cluster import (
    Cluster,
    Cluster_Settings,
    ReferenceClockSource,
)
from qibolab.instruments.qblox.cluster_qcm_bb import ClusterQCM_BB
from qibolab.instruments.qblox.cluster_qcm_rf import ClusterQCM_RF
from qibolab.instruments.qblox.cluster_qrm_rf import ClusterQRM_RF
from qibolab.instruments.qblox.controller import QbloxController
from qibolab.instruments.rohde_schwarz import SGS100A
from qibolab.platform import Platform
from qibolab.serialize import (
    load_instrument_settings,
    load_qubits,
    load_runcard,
    load_settings,
)

NAME = "qblox"
ADDRESS = "192.168.0.6"
TIME_OF_FLIGHT = 500
RUNCARD = pathlib.Path(__file__).parent / "qblox.yml"


def create(runcard_path=RUNCARD):
    """QuantWare 5q-chip controlled using qblox cluster.

    Args:
        runcard_path (str): Path to the runcard file.
    """

    runcard = load_runcard(runcard_path)
    modules = {}

    cluster = Cluster(
        name="cluster",
        address=ADDRESS,
        settings=Cluster_Settings(reference_clock_source=ReferenceClockSource.INTERNAL),
    )

    # DEBUG: debug folder = report folder
    # import os
    # folder = os.path.dirname(runcard) + "/debug/"
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # for name in modules:
    #     modules[name]._debug_folder = folder
    qcm_bb0 = ClusterQCM_BB("qcm_bb0", f"{ADDRESS}:2", cluster)
    qcm_bb1 = ClusterQCM_BB("qcm_bb1", f"{ADDRESS}:4", cluster)
    qcm_rf0 = ClusterQCM_RF("qcm_rf0", f"{ADDRESS}:6", cluster)
    qcm_rf1 = ClusterQCM_RF("qcm_rf1", f"{ADDRESS}:8", cluster)
    qcm_rf2 = ClusterQCM_RF("qcm_rf2", f"{ADDRESS}:10", cluster)
    qrm_rf_a = ClusterQRM_RF("qrm_rf_a", f"{ADDRESS}:16", cluster)
    qrm_rf_b = ClusterQRM_RF("qrm_rf_b", f"{ADDRESS}:18", cluster)

    controller = QbloxController("qblox_controller", cluster, modules)

    twpa_pump = SGS100A(name="twpa_pump", address="192.168.0.36")
    instruments = {
        controller.name: controller,
        twpa_pump.name: twpa_pump,
        qcm_bb0.name: qcm_bb0,
        qcm_bb1.name: qcm_bb1,
        qcm_rf0.name: qcm_rf0,
        qcm_rf1.name: qcm_rf1,
        qcm_rf2.name: qcm_rf2,
        qrm_rf_a.name: qrm_rf_a,
        qrm_rf_b.name: qrm_rf_b,
    }

    instruments = load_instrument_settings(runcard, instruments)

    modules = {
        name: instrument
        for name, instrument in instruments.items()
        if isinstance(instrument, (ClusterQCM_RF, ClusterQCM_BB, ClusterQRM_RF))
    }

    # Create channel objects
    channels = {}
    # Readout
    channels["L3-25_a"] = Channel(name="L3-25_a", port=qrm_rf_a.ports["o1"])
    channels["L3-25_b"] = Channel(name="L3-25_b", port=qrm_rf_b.ports["o1"])
    # Feedback
    channels["L2-5_a"] = Channel(name="L2-5_a", port=qrm_rf_a.ports["i1"])
    channels["L2-5_b"] = Channel(name="L2-5_b", port=qrm_rf_b.ports["i1"])
    # Drive
    channels["L3-15"] = Channel(name="L3-15", port=qcm_rf0.ports["o1"])
    channels["L3-11"] = Channel(name="L3-11", port=qcm_rf0.ports["o2"])
    channels["L3-12"] = Channel(name="L3-12", port=qcm_rf1.ports["o1"])
    channels["L3-13"] = Channel(name="L3-13", port=qcm_rf1.ports["o2"])
    channels["L3-14"] = Channel(name="L3-14", port=qcm_rf2.ports["o1"])
    # Flux
    channels["L4-5"] = Channel(name="L4-5", port=qcm_bb0.ports["o1"])
    channels["L4-1"] = Channel(name="L4-1", port=qcm_bb0.ports["o2"])
    channels["L4-2"] = Channel(name="L4-2", port=qcm_bb0.ports["o3"])
    channels["L4-3"] = Channel(name="L4-3", port=qcm_bb0.ports["o4"])
    channels["L4-4"] = Channel(name="L4-4", port=qcm_bb1.ports["o1"])
    # TWPA
    channels["L3-28"] = Channel(name="L3-28", port=None)
    channels["L3-28"].local_oscillator = twpa_pump

    # create qubit objects

    qubits, couplers, pairs = load_qubits(runcard)
    # remove witness qubit
    # del qubits[5]
    # assign channels to qubits
    for q in [0, 1]:
        qubits[q].readout = channels["L3-25_a"]
        qubits[q].feedback = channels["L2-5_a"]
        qubits[q].twpa = channels["L3-28"]
    for q in [2, 3, 4]:
        qubits[q].readout = channels["L3-25_b"]
        qubits[q].feedback = channels["L2-5_b"]
        qubits[q].twpa = channels["L3-28"]

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

    settings = load_settings(runcard)

    return Platform("qblox", qubits, pairs, instruments, settings, resonator_type="2D")
