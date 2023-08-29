import itertools
import pathlib

import laboneq.simple as lo

from qibolab.channels import Channel, ChannelMap
from qibolab.instruments.oscillator import LocalOscillator
from qibolab.instruments.zhinst import Zurich
from qibolab.platform import Platform
from qibolab.serialize import (
    load_couplers,
    load_qubits,
    load_runcard,
    load_settings,
    register_gates,
)

RUNCARD = pathlib.Path(__file__).parent / "zurich.yml"


# Function returning a calibrated device setup
def create_descriptor():
    """
    Function returning a device setup
    """

    # Instantiate Zh set of instruments[They work as one]
    instruments = {
        "SHFQC": [{"address": "DEV12146", "uid": "device_shfqc"}],
        "HDAWG": [
            {"address": "DEV8660", "uid": "device_hdawg"},
            {"address": "DEV8673", "uid": "device_hdawg2"},
        ],
        "PQSC": [{"address": "DEV10055", "uid": "device_pqsc"}],
    }

    shfqc = []
    for i in range(5):
        shfqc.append({"iq_signal": f"q{i}/drive_line", "ports": f"SGCHANNELS/{i}/OUTPUT"})
        shfqc.append({"iq_signal": f"q{i}/measure_line", "ports": ["QACHANNELS/0/OUTPUT"]})
        shfqc.append({"acquire_signal": f"q{i}/acquire_line", "ports": ["QACHANNELS/0/INPUT"]})

    hdawg = []
    for i in range(5):
        hdawg.append({"rf_signal": f"q{i}/flux_line", "ports": f"SIGOUTS/{i}"})
    for c, i in zip(itertools.chain(range(0, 2), range(3, 4)), range(5, 8)):
        hdawg.append({"rf_signal": f"qc{c}/flux_line", "ports": f"SIGOUTS/{i}"})

    hdawg2 = [{"rf_signal": "qc4/flux_line", "ports": f"SIGOUTS/0"}]

    pqsc = [
        "internal_clock_signal",
        {"to": "device_hdawg2", "port": "ZSYNCS/4"},
        {"to": "device_hdawg", "port": "ZSYNCS/2"},
        {"to": "device_shfqc", "port": "ZSYNCS/0"},
    ]

    connections = {
        "device_shfqc": shfqc,
        "device_hdawg": hdawg,
        "device_hdawg2": hdawg2,
        "device_pqsc": pqsc,
    }

    descriptor = {
        "instruments": instruments,
        "connections": connections,
    }

    return descriptor


def create(runcard_path=RUNCARD):
    """Create platform using Zurich Instrumetns (Zh) SHFQC, HDAWGs and PQSC.

    Based on IQM 5-qubit chip.
    Instrument related parameters are hardcoded in ``__init__`` and ``setup``.

    Args:
        runcard (str): Path to the runcard file.
    """
    descriptor = create_descriptor()

    controller = Zurich("EL_ZURO", descriptor, use_emulation=False, time_of_flight=280, smearing=100)

    # Create channel objects and map controllers
    channels = ChannelMap()
    # readout
    channels |= Channel("L3-31", port=controller[("device_shfqc", "[QACHANNELS/0/INPUT]")])
    # feedback
    channels |= Channel("L2-7", port=controller[("device_shfqc", "[QACHANNELS/0/OUTPUT]")])
    # drive
    channels |= (
        Channel(f"L4-{i}", port=controller[("device_shfqc", f"SGCHANNELS/{i-5}/OUTPUT")]) for i in range(15, 20)
    )
    # flux qubits (CAREFUL WITH THIS !!!)
    channels |= (Channel(f"L4-{i}", port=controller[("device_hdawg", f"SIGOUTS/{i-6}")]) for i in range(6, 11))
    # flux couplers
    channels |= (Channel(f"L4-{i}", port=controller[("device_hdawg", f"SIGOUTS/{i-11+5}")]) for i in range(11, 14))
    channels |= Channel("L4-14", port=controller[("device_hdawg2", f"SIGOUTS/0")])

    # SHFQC
    # Sets the maximal Range of the Signal Output power.
    # The instrument selects the closest available Range with a resolution of 5 dBm.

    # feedback
    channels["L3-31"].power_range = 10
    # readout
    channels["L2-7"].power_range = -25
    # drive
    for i in range(5, 10):
        channels[f"L4-1{i}"].power_range = -10

    # HDAWGS
    # Sets the output voltage range.
    # The instrument selects the next higher available Range with a resolution of 0.4 Volts.

    # flux
    for i in range(6, 11):
        channels[f"L4-{i}"].power_range = 0.8
    # flux couplers
    for i in range(11, 15):
        channels[f"L4-{i}"].power_range = 0.8

    # Instantiate local oscillators
    local_oscillators = [LocalOscillator(f"lo_{kind}", None) for kind in ["readout"] + [f"drive_{n}" for n in range(4)]]

    # Set Dummy LO parameters (Map only the two by two oscillators)
    local_oscillators[0].frequency = 5_500_000_000  # For SG0 (Readout)
    local_oscillators[1].frequency = 4_200_000_000  # For SG1 and SG2 (Drive)
    local_oscillators[2].frequency = 4_600_000_000  # For SG3 and SG4 (Drive)
    local_oscillators[3].frequency = 4_800_000_000  # For SG5 and SG6 (Drive)

    # Map LOs to channels
    ch_to_lo = {"L2-7": 0, "L4-15": 1, "L4-16": 1, "L4-17": 2, "L4-18": 2, "L4-19": 3}
    for ch, lo in ch_to_lo.items():
        channels[ch].local_oscillator = local_oscillators[lo]

    # create qubit objects from runcard
    runcard = load_runcard(runcard_path)
    qubits, pairs = load_qubits(runcard)
    couplers, coupler_pairs = load_couplers(runcard)
    settings = load_settings(runcard)

    # assign channels to qubits and sweetspots(operating points)
    for q in range(0, 5):
        qubits[q].feedback = channels["L3-31"]
        qubits[q].readout = channels["L2-7"]

    for q in range(0, 5):
        qubits[q].drive = channels[f"L4-{15 + q}"]
        qubits[q].flux = channels[f"L4-{6 + q}"]
        channels[f"L4-{6 + q}"].qubit = qubits[q]

    # assign channels to couplers and sweetspots(operating points)
    for c, coupler in couplers.items():
        coupler.flux = channels[f"L4-{11 + c}"]
        # Is this needed ?
        # channels[f"L4-{11 + c}"].qubit = qubits[f"c{c}"]

    # FIXME: Call couplers by its name
    # assign couplers to qubits
    for c in itertools.chain(range(0, 2), range(3, 5)):
        qubits[c].flux_coupler[c] = couplers[c].name
        qubits[2].flux_coupler[c] = couplers[c].name

    qubits, pairs = register_gates(runcard, qubits, pairs, couplers)
    instruments = {controller.name: controller}
    instruments.update({lo.name: lo for lo in local_oscillators})
    settings = load_settings(runcard)
    return Platform("zurich", qubits, pairs, instruments, settings, resonator_type="2D")
