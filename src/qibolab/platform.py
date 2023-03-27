import networkx as nx
from qibo.config import raise_error

from qibolab.designs import Channel, ChannelMap, InstrumentDesign
from qibolab.platforms.platform import DesignPlatform


def create_dummy(runcard):
    """Create a dummy platform using the dummy instrument.

    Useful for testing.
    """
    from qibolab.instruments.dummy import DummyInstrument

    # Create channel objects
    channels = ChannelMap()
    channels |= ChannelMap.from_names("readout", "drive", "flux")

    # Create dummy controller
    instrument = DummyInstrument("dummy", 0)
    # Create design
    design = InstrumentDesign([instrument], channels)
    # Create platform
    platform = DesignPlatform("dummy", design, runcard)

    # map channels to qubits
    for qubit in platform.qubits:
        platform.qubits[qubit].readout = channels["readout"]
        platform.qubits[qubit].drive = channels["drive"]
        platform.qubits[qubit].flux = channels["flux"]

    return platform


def create_tii_qw25q_v3_e10(runcard, simulation_duration=None, address=None, cloud=False):
    """Create platform using Quantum Machines (QM) OPXs and Rohde Schwarz/ERAsynth local oscillators.

    IPs and other instrument related parameters are hardcoded in ``__init__`` and ``setup``.

    Args:
        runcard (str): Path to the runcard file.
        simulation_duration (int): Duration for the simulation in ns.
            If given the compiler simulator will be used instead of the actual hardware.
            Default is ``None`` which falls back to the hardware.
        address (str): Address and port for the QM OPX cluster.
            If ``None`` it will attempt to connect to TII instruments.
        cloud (bool): See :class:`qibolab.instruments.qmsim.QMSim` for details.
            Relevant only when ``simulation_duration`` is given.
    """
    # Create channel objects
    channels = ChannelMap()

    # Wiring
    wiring = {
        "feedback": {
            "A": ["L2-1_a", "L2-1_b"],
            "B": ["L2-2_a", "L2-2_b"],
            "C": ["L2-3_a", "L2-3_b"],
            "D": ["L2-4_a", "L2-4_b"],
        },
        "readout": {
            "A": ["L3-26_a", "L3-26_b"],
            "B": ["L3-27_a", "L3-27_b"],
            "C": ["L3-18_a", "L3-18_b"],
            "D": ["L3-30_a", "L3-30_b"],
        },
        "drive": {
            "A": [f"L3-{i}" for i in range(1, 7)],
            "B": [f"L3-{i}" for i in range(7, 10)] + ["L3-19", "L4-22"],
            "C": [f"L4-{i}" for i in range(23, 28)],
            "D": [f"L4-{i}" for i in range(28, 31)],
        },
        "flux": {
            "A": [f"L1-{i}" for i in range(5, 10)] + ["L1-4"],
            "B": [f"L1-{i}" for i in range(10, 16)],
            "C": [f"L1-{i}" for i in range(16, 21)],
            "D": [f"L1-{i}" for i in range(21, 26)],
        },
    }
    
    connections = {
        "A": [1,2,3],
        "B": [4,5],
        "C": [6,7],
        "D": [8,9],
    }

    # Create channels
    for channel in wiring:
        for qubit in wiring[qubit]:
            for wire in wiring[qubit][channel]:
                channels |= ChannelMap.from_names(wire)

    for qubit in connections:
        channels[wiring["feedback"][qubit][0]].port = [(f"con{connections[qubit][0]}", 1), (f"con{connections[qubit][0]}", 2)]
        channels[wiring["feedback"][qubit][1]].port = [(f"con{connections[qubit][1]}", 1), (f"con{connections[qubit][1]}", 2)]
        channels[wiring["readout"][qubit][0]].port = [(f"con{connections[qubit][0]}", 9), (f"con{connections[qubit][0]}", 10)]
        channels[wiring["readout"][qubit][1]].port = [(f"con{connections[qubit][1]}", 9), (f"con{connections[qubit][1]}", 10)]

        wires_list = wiring["drive"][qubit]
        for i in range(len(wires_list)):
            channels[wires_list[i]].port = [(f"con{connections[qubit][(2*i)//8]}", 2*i%8 + 1), (f"con{connections[qubit][(2*i)//8]}", 2*i%8 + 2)]
            last_port = 2*i%8 + 2

        wires_list = wiring["flux"][qubit]
        for i in range(len(wires_list)):
            channels[wires_list[i]].port = [(f"con{connections[qubit][i//8]}", (i+last_port)%8 + 1)] 

    # Instantiate QM OPX instruments
    if simulation_duration is None:
        from qibolab.instruments.qm import QMOPX
        from qibolab.instruments.rohde_schwarz import SGS100A as LocalOscillator

        controller = QMOPX("qmopx", "192.168.0.1:80")

    else:
        from qibolab.instruments.dummy_oscillator import (
            DummyLocalOscillator as LocalOscillator,
        )
        from qibolab.instruments.qmsim import QMSim

        if address is None:
            # connect to TII instruments for simulation
            address = "192.168.0.1:80"

        controller = QMSim("qmopx", address, simulation_duration, cloud)

    # set time of flight for readout integration (HARDCODED)
    controller.time_of_flight = 280

    # Instantiate local oscillators (HARDCODED)
    local_oscillators = [LocalOscillator(f"era_0{i}", f"192.168.0.20{i}") for i in range(1, 9)] + \
        [LocalOscillator(f"LO_{i}", f"192.168.0.3{i}") for i in [1,2,3,4,5,6,9]]
    drive_local_oscillators = {
        "A": [LocalOscillator(f"LO_05", f"192.168.0.35")] + \
            2*[LocalOscillator(f"LO_01", f"192.168.0.31")] + \
            2*[LocalOscillator(f"LO_01", f"192.168.0.31")]+\
            [LocalOscillator("era_01", "192.168.0.201")],
        "B": [LocalOscillator(f"era_02", f"192.168.0.202")] + \
            4*[LocalOscillator(f"LO_06", f"192.168.0.206")],
        "C": [LocalOscillator(f"era_0{i}", f"192.168.0.20{i}") for i in range(3, 8)],
        "D": [LocalOscillator(f"era_08", "192.168.0.208")] + \
            2*[LocalOscillator(f"LO_01", "192.168.0.31")]
    }

    instruments = [controller] + local_oscillators
    design = InstrumentDesign(instruments, channels)
    platform = DesignPlatform("qw25q", design, runcard)

    # assign channels to qubits
    qubits = platform.qubits
    for channel in ["flux", "drive"]:
        for qubit in wiring[channel]:
            for i, wire in enumerate(wiring[channel][qubit]):
                q = f"{qubit}{i+1}"
                if channel == "flux":
                    qubits[q].flux = wire
                elif channel == "drive":
                    qubits[q].drive = wire
                    channels[wire].local_oscillator = drive_local_oscillators[qubit][i]
    
    for q in range(0, 4):
        for qubit in connections:
            qubits[f"{qubit}{q}"].readout = wiring["readout"][qubit][0]
            qubits[f"{qubit}{q}"].feedback = wiring["feedback"][qubit][0]
    for q in range(4, 7):
        for qubit in connections:
            if (not qubit == "A") and (q == 6):
                break
            qubits[f"{qubit}{q}"].readout = wiring["readout"][qubit][1]
            qubits[f"{qubit}{q}"].feedback = wiring["feedback"][qubit][1]
    
    # Add LO_03 to all readout lines on their first module
    for qubit in connections:
        channels[wiring["readout"][qubit][0]].local_oscillator = LocalOscillator("LO_03", "192.168.0.33")
        channels[wiring["readout"][qubit][1]].local_oscillator = LocalOscillator("LO_04", "192.168.0.34")
        channels[wiring["feedback"][qubit][0]].local_oscillator = LocalOscillator("LO_03", "192.168.0.33")
        channels[wiring["feedback"][qubit][1]].local_oscillator = LocalOscillator("LO_04", "192.168.0.34")
    
    # Configure local oscillator's frequency and power
    for q in qubits:
        if qubits[q].readout.local_oscillator.name == "LO_01":
            qubits[q].readout.local_oscillator.frequency = 6e9
            qubits[q].readout.local_oscillator.power = 21
        elif qubits[q].readout.local_oscillator.name == "LO_03":
            qubits[q].readout.local_oscillator.frequency = 6.5e9
            qubits[q].readout.local_oscillator.power = 21
        elif qubits[q].readout.local_oscillator.name == "LO_04":
            qubits[q].readout.local_oscillator.frequency = 6.5e9
            qubits[q].readout.local_oscillator.power = 21
        elif qubits[q].readout.local_oscillator.name == "LO_05":
            qubits[q].readout.local_oscillator.frequency = 6e9
            qubits[q].readout.local_oscillator.power = 18
        elif qubits[q].readout.local_oscillator.name == "LO_06":
            qubits[q].readout.local_oscillator.frequency = 6e9
            qubits[q].readout.local_oscillator.power = 21
        
        qubits[q].readout.local_oscillator.frequency = 6.5e9
    

    # Platfom topology
    Q = [chr(i+65) + str(j+1) for i in range(4) for j in range(5)] + "A6"
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [
        (Q[0], Q[2]),
        (Q[1], Q[2]),
        (Q[3], Q[2]),
        (Q[4], Q[2]),
    ]
    chip.add_edges_from(graph_list)
    platform.topology = chip

    return platform


def Platform(name, runcard=None, design=None):
    """Platform for controlling quantum devices.

    Args:
        name (str): name of the platform. Options are 'tiiq', 'qili' and 'icarusq'.
        runcard (str): path to the yaml file containing the platform setup.
        design (:class:`qibolab.designs.abstract.AbstractInstrumentDesign`): Instrument
            design to use for the platform.

    Returns:
        The plaform class.
    """
    if not runcard:
        from os.path import exists

        from qibolab.paths import qibolab_folder

        runcard = qibolab_folder / "runcards" / f"{name}.yml"
        if not exists(runcard):
            raise_error(RuntimeError, f"Runcard {name} does not exist.")

    if name == "dummy":
        return create_dummy(runcard)
    elif name == "icarusq":
        from qibolab.platforms.icplatform import ICPlatform as Device
    elif name == "qw5q_gold":
        return create_tii_qw5q_gold(runcard)
    else:
        from qibolab.platforms.multiqubit import MultiqubitPlatform as Device

    return Device(name, runcard)
