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

    modules["qcm_bb0"] = qcm_bb0
    modules["qcm_bb1"] = qcm_bb1
    modules["qcm_rf0"] = qcm_rf0
    modules["qcm_rf1"] = qcm_rf1
    modules["qcm_rf2"] = qcm_rf2
    modules["qrm_rf_a"] = qrm_rf_a
    modules["qrm_rf_b"] = qrm_rf_b

    # Create channel objects
    channels = {}

    def instantiate_channels(channel_name: str, module, module_port_name: str):
        module.channels.append(channel_name)
        module._port_channel_map[module_port_name] = channel_name
        module._channel_port_map[channel_name] = module_port_name
        return Channel(name=channel_name, port=module.ports[module_port_name])

    # readout
    channels["L3-25_a"] = instantiate_channels(channel_name="L3-25_a", module=qrm_rf_a, module_port_name="o1")
    channels["L3-25_b"] = instantiate_channels(channel_name="L3-25_b", module=qrm_rf_b, module_port_name="o1")
    # feedback
    channels["L2-5_a"] = instantiate_channels(channel_name="L2-5_a", module=qrm_rf_a, module_port_name="i1")
    channels["L2-5_b"] = instantiate_channels(channel_name="L2-5_b", module=qrm_rf_b, module_port_name="i1")
    # drive
    channels["L3-15"] = instantiate_channels(channel_name="L3-15", module=qcm_rf0, module_port_name="o1")
    channels["L3-11"] = instantiate_channels(channel_name="L3-11", module=qcm_rf0, module_port_name="o2")
    channels["L3-12"] = instantiate_channels(channel_name="L3-12", module=qcm_rf1, module_port_name="o1")
    channels["L3-13"] = instantiate_channels(channel_name="L3-13", module=qcm_rf1, module_port_name="o2")
    channels["L3-14"] = instantiate_channels(channel_name="L3-14", module=qcm_rf2, module_port_name="o1")
    # flux
    channels["L4-5"] = instantiate_channels(channel_name="L4-5", module=qcm_bb0, module_port_name="o1")
    channels["L4-1"] = instantiate_channels(channel_name="L4-1", module=qcm_bb0, module_port_name="o2")
    channels["L4-2"] = instantiate_channels(channel_name="L4-2", module=qcm_bb0, module_port_name="o3")
    channels["L4-3"] = instantiate_channels(channel_name="L4-3", module=qcm_bb0, module_port_name="o4")
    channels["L4-4"] = instantiate_channels(channel_name="L4-4", module=qcm_bb1, module_port_name="o1")
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
