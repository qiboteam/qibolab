import itertools
import pathlib

from laboneq.dsl.device import create_connection
from laboneq.dsl.device.instruments import HDAWG, PQSC, SHFQC
from laboneq.simple import DeviceSetup

from qibolab import Platform
from qibolab.channels import Channel, ChannelMap
from qibolab.instruments.dummy import DummyLocalOscillator as LocalOscillator
from qibolab.instruments.zhinst import Zurich
from qibolab.serialize import (
    load_instrument_settings,
    load_qubits,
    load_runcard,
    load_settings,
)

RUNCARD = pathlib.Path(__file__).parent / "zurich.yml"
FOLDER = pathlib.Path(__file__).parent / "iqm5q/"
N_QUBITS = 5


def create(runcard_path=RUNCARD):
    """IQM 5q-chip controlled Zurich Instrumetns (Zh) SHFQC, HDAWGs and PQSC.

    Args:
        runcard_path (str): Path to the runcard file.
    """

    device_setup = DeviceSetup("EL_ZURO")
    # Dataserver
    device_setup.add_dataserver(host="localhost", port=8004)
    # Instruments
    device_setup.add_instruments(
        HDAWG("device_hdawg", address="DEV8660"),
        HDAWG("device_hdawg2", address="DEV8673"),
        PQSC("device_pqsc", address="DEV10055", reference_clock_source="internal"),
        SHFQC("device_shfqc", address="DEV12146"),
    )
    device_setup.add_connections(
        "device_shfqc",
        *[
            create_connection(
                to_signal=f"q{i}/drive_line", ports=[f"SGCHANNELS/{i}/OUTPUT"]
            )
            for i in range(N_QUBITS)
        ],
        *[
            create_connection(
                to_signal=f"q{i}/measure_line", ports=["QACHANNELS/0/OUTPUT"]
            )
            for i in range(N_QUBITS)
        ],
        *[
            create_connection(
                to_signal=f"q{i}/acquire_line", ports=["QACHANNELS/0/INPUT"]
            )
            for i in range(N_QUBITS)
        ],
    )
    device_setup.add_connections(
        "device_hdawg",
        *[
            create_connection(to_signal=f"q{i}/flux_line", ports=f"SIGOUTS/{i}")
            for i in range(N_QUBITS)
        ],
        *[
            create_connection(to_signal=f"qc{c}/flux_line", ports=f"SIGOUTS/{i}")
            for c, i in zip(itertools.chain(range(0, 2), range(3, 4)), range(5, 8))
        ],
    )

    device_setup.add_connections(
        "device_hdawg2",
        create_connection(to_signal="qc4/flux_line", ports=["SIGOUTS/0"]),
    )

    device_setup.add_connections(
        "device_pqsc",
        create_connection(to_instrument="device_hdawg2", ports="ZSYNCS/1"),
        create_connection(to_instrument="device_hdawg", ports="ZSYNCS/0"),
        create_connection(to_instrument="device_shfqc", ports="ZSYNCS/2"),
    )

    controller = Zurich(
        "EL_ZURO",
        device_setup=device_setup,
        use_emulation=False,
        time_of_flight=75,
        smearing=50,
    )

    # Create channel objects and map controllers
    channels = ChannelMap()
    # feedback
    channels |= Channel(
        "L2-7", port=controller[("device_shfqc", "[QACHANNELS/0/INPUT]")]
    )
    # readout
    channels |= Channel(
        "L3-31", port=controller[("device_shfqc", "[QACHANNELS/0/OUTPUT]")]
    )
    # drive
    channels |= (
        Channel(
            f"L4-{i}", port=controller[("device_shfqc", f"SGCHANNELS/{i-5}/OUTPUT")]
        )
        for i in range(15, 20)
    )
    # flux qubits (CAREFUL WITH THIS !!!)
    channels |= (
        Channel(f"L4-{i}", port=controller[("device_hdawg", f"SIGOUTS/{i-6}")])
        for i in range(6, 11)
    )
    # flux couplers
    channels |= (
        Channel(f"L4-{i}", port=controller[("device_hdawg", f"SIGOUTS/{i-11+5}")])
        for i in range(11, 14)
    )
    channels |= Channel("L4-14", port=controller[("device_hdawg2", "SIGOUTS/0")])
    # TWPA pump(EraSynth)
    channels |= Channel("L3-32")

    # SHFQC
    # Sets the maximal Range of the Signal Output power.
    # The instrument selects the closest available Range [-50. -30. -25. -20. -15. -10.  -5.   0.   5.  10.]
    # with a resolution of 5 dBm.

    # readout "gain": Set to max power range (10 Dbm) if no distorsion
    channels["L3-31"].power_range = -15  # -15
    # feedback "gain": play with the power range to calibrate the best RO
    channels["L2-7"].power_range = 10

    # drive
    # The instrument selects the closest available Range [-30. -25. -20. -15. -10.  -5.   0.   5.  10.]
    channels["L4-15"].power_range = -10  # q0
    channels["L4-16"].power_range = -5  # q1
    channels["L4-17"].power_range = -10  # q2
    channels["L4-18"].power_range = -5  # q3
    channels["L4-19"].power_range = -10  # q4

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
    local_oscillators = [
        LocalOscillator(f"lo_{kind}", None)
        for kind in ["readout"] + [f"drive_{n}" for n in range(3)]
    ]

    # Map LOs to channels
    ch_to_lo = {
        "L3-31": 0,
        "L4-15": 1,
        "L4-16": 1,
        "L4-17": 2,
        "L4-18": 2,
        "L4-19": 3,
    }
    for ch, lo in ch_to_lo.items():
        channels[ch].local_oscillator = local_oscillators[lo]

    # create qubit objects
    runcard = load_runcard(runcard_path)
    qubits, couplers, pairs = load_qubits(runcard, FOLDER)
    settings = load_settings(runcard)

    # assign channels to qubits and sweetspots(operating points)
    for q in range(0, 5):
        qubits[q].readout = channels["L3-31"]
        qubits[q].feedback = channels["L2-7"]

    for q in range(0, 5):
        qubits[q].drive = channels[f"L4-{15 + q}"]
        qubits[q].flux = channels[f"L4-{6 + q}"]
        qubits[q].twpa = channels["L3-32"]
        channels[f"L4-{6 + q}"].qubit = qubits[q]

    # assign channels to couplers and sweetspots(operating points)
    for c, coupler in enumerate(couplers.values()):
        coupler.flux = channels[f"L4-{11 + c}"]
    instruments = {controller.name: controller}
    instruments.update({lo.name: lo for lo in local_oscillators})
    instruments = load_instrument_settings(runcard, instruments)
    return Platform(
        "zurich",
        qubits,
        pairs,
        instruments,
        settings,
        resonator_type="2D",
        couplers=couplers,
    )
