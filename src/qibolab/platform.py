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


def create_tii_qw25q(runcard, simulation_duration=None, address=None, cloud=False):
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
            # "D": ["L2-4_a", "L2-4_b"],
        },
        "readout": {
            "A": ["L3-26_a", "L3-26_b"],
            "B": ["L3-27_a", "L3-27_b"],
            "C": ["L3-18_a", "L3-18_b"],
            # "D": ["L3-30_a", "L3-30_b"],
        },
        "drive": {
            "A": [f"L3-{i}" for i in range(1, 7)],
            "B": [f"L3-{i}" for i in range(7, 10)] + ["L3-19", "L4-22"],
            "C": [f"L4-{i}" for i in range(23, 28)],
            # "D": [f"L4-{i}" for i in range(28, 31)],
        },
        "flux": {
            "A": [f"L1-{i}" for i in range(5, 10)] + ["L1-4"],
            "B": [f"L1-{i}" for i in range(11, 16)],
            "C": [f"L1-{i}" for i in range(16, 21)],
            # "D": [f"L1-{i}" for i in range(21, 26)],
        },
    }

    connections = {
        "A": [1, 2, 3],
        "B": [4, 5],
        "C": [6, 7],
        # "D": [8, 9],
    }

    # Create channels
    for channel in wiring:
        for feedline in wiring[channel]:
            for wire in wiring[channel][feedline]:
                channels |= ChannelMap.from_names(wire)

    for feedline in connections:
        channels[wiring["feedback"][feedline][0]].ports = [
            (f"con{connections[feedline][0]}", 1),
            (f"con{connections[feedline][0]}", 2),
        ]
        channels[wiring["feedback"][feedline][1]].ports = [
            (f"con{connections[feedline][1]}", 1),
            (f"con{connections[feedline][1]}", 2),
        ]
        channels[wiring["readout"][feedline][0]].ports = [
            (f"con{connections[feedline][0]}", 10),
            (f"con{connections[feedline][0]}", 9),
        ]
        channels[wiring["readout"][feedline][1]].ports = [
            (f"con{connections[feedline][1]}", 10),
            (f"con{connections[feedline][1]}", 9),
        ]

        wires_list = wiring["drive"][feedline]
        for i in range(len(wires_list)):
            channels[wires_list[i]].ports = [
                (f"con{connections[feedline][(2*i)//8]}", 2 * i % 8 + 1),
                (f"con{connections[feedline][(2*i)//8]}", 2 * i % 8 + 2),
            ]
            last_port = 2 * i % 8 + 2
            last_con = (2 * i) // 8

        wires_list = wiring["flux"][feedline]
        for i in range(len(wires_list)):
            channels[wires_list[i]].ports = [
                (f"con{connections[feedline][last_con + (i + last_port)//8]}", (i + last_port) % 8 + 1)
            ]

    # Instantiate QM OPX instruments
    if simulation_duration is None:
        from qibolab.instruments.erasynth import ERA as LO_ERA
        from qibolab.instruments.qm import QMOPX
        from qibolab.instruments.rohde_schwarz import SGS100A as LO_RS_SGS100A

        controller = QMOPX("qmopx", "192.168.0.101:80")

    else:
        from qibolab.instruments.dummy_oscillator import DummyLocalOscillator as LO_ERA
        from qibolab.instruments.dummy_oscillator import (
            DummyLocalOscillator as LO_RS_SGS100A,
        )
        from qibolab.instruments.qmsim import QMSim

        if address is None:
            # connect to TII instruments for simulation
            address = "192.168.0.1:80"

        controller = QMSim("qmopx", address, simulation_duration, cloud)

    # set time of flight for readout integration (HARDCODED)
    controller.time_of_flight = 280

    # Instantiate local oscillators (HARDCODED)
    local_oscillators = [LO_ERA(f"era_0{i}", f"192.168.0.20{i}") for i in range(1, 9)] + [
        LO_RS_SGS100A(f"LO_0{i}", f"192.168.0.3{i}") for i in [1, 3, 4, 5, 6, 9]
    ]
    drive_local_oscillators = {
        "A": ["LO_05"] + 2 * ["LO_01"] + ["LO_05"] + ["LO_01"] + ["era_01"],
        "B": ["era_02"] + 4 * ["LO_06"],
        "C": [f"era_0{i}" for i in range(3, 8)],
        # "D": ["era_08"] + 2 * ["LO_01"],
    }
    # Configure local oscillator's frequency and power
    for lo in local_oscillators:
        if lo.name == "LO_01":
            lo.frequency = 6.15e9
            lo.power = 21
        elif lo.name == "LO_04":
            lo.frequency = 7.1e9
            lo.power = 23
        elif lo.name == "LO_03":
            lo.frequency = 7.8e9
            lo.power = 23
        elif lo.name == "LO_05":
            lo.frequency = 5.37e9
            lo.power = 18
        elif lo.name == "LO_06":
            lo.frequency = 6.2e9
            lo.power = 21
        elif "era" in lo.name:
            lo.frequency = 4e9
            lo.power = 15

    # Assign local oscillators to channels
    for lo in local_oscillators:
        if lo.name == "LO_03":
            for feedline in connections:
                channels[wiring["readout"][feedline][0]].local_oscillator = lo
                channels[wiring["feedback"][feedline][0]].local_oscillator = lo
        elif lo.name == "LO_04":
            for feedline in connections:
                channels[wiring["readout"][feedline][1]].local_oscillator = lo
                channels[wiring["feedback"][feedline][1]].local_oscillator = lo
        else:
            for feedline in drive_local_oscillators:
                for i, name in enumerate(drive_local_oscillators[feedline]):
                    if lo.name == name:
                        channels[wiring["drive"][feedline][i]].local_oscillator = lo

    instruments = [controller] + local_oscillators
    design = InstrumentDesign(instruments, channels)
    platform = DesignPlatform("qw25q", design, runcard)

    # assign channels to qubits
    qubits = platform.qubits
    for channel in ["flux", "drive"]:
        for feedline in wiring[channel]:
            for i, wire in enumerate(wiring[channel][feedline]):
                q = f"{feedline}{i+1}"
                if channel == "flux":
                    qubits[q].flux = channels[wire]
                    channels[wire].qubit = qubits[q]
                elif channel == "drive":
                    qubits[q].drive = channels[wire]
                    if "era" in qubits[q].drive.local_oscillator.name:
                        qubits[q].drive.local_oscillator.frequency = qubits[q].drive_frequency + 200e6

    for q in ["A3", "A5", "A6", "B4", "B5", "C2", "C3", "C5"]:  # Qubits with LO around 7e9
        qubits[q].readout = channels[wiring["readout"][q[0]][0]]
        qubits[q].feedback = channels[wiring["feedback"][q[0]][0]]
    for q in ["A1", "A2", "A4", "B1", "B2", "B3", "C1", "C4"]:  # Qubits with LO around 7.5e9
        qubits[q].readout = channels[wiring["readout"][q[0]][1]]
        qubits[q].feedback = channels[wiring["feedback"][q[0]][1]]

    # Save temporarely the qubits as a yaml file
    import yaml

    yaml.dump(qubits, open("qubits.yaml", "w"))

    # Platfom topology
    Q = []
    for i in range(1, 7):
        Q += ["A{i}"]
    for i in range(1, 6):
        Q += ["B{i}"]
    for i in range(1, 6):
        Q += ["C{i}"]
    for i in range(1, 6):
        Q += ["D{i}"]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [
        (Q[0], Q[1]),
        (Q[0], Q[2]),
        (Q[0], Q[20]),
        (Q[1], Q[3]),
        (Q[2], Q[4]),
        (Q[2], Q[19]),
        (Q[3], Q[4]),
        (Q[3], Q[8]),
        (Q[4], Q[6]),
        (Q[5], Q[2]),
        (Q[5], Q[8]),
        (Q[5], Q[18]),
        (Q[5], Q[13]),
        (Q[6], Q[8]),
        (Q[6], Q[7]),
        (Q[7], Q[9]),
        (Q[8], Q[9]),
        (Q[9], Q[10]),
        (Q[9], Q[13]),
        (Q[10], Q[11]),
        (Q[11], Q[13]),
        (Q[11], Q[12]),
        (Q[12], Q[14]),
        (Q[13], Q[14]),
        (Q[14], Q[18]),
        (Q[14], Q[15]),
        (Q[15], Q[16]),
        (Q[16], Q[18]),
        (Q[16], Q[17]),
        (Q[17], Q[19]),
        (Q[18], Q[19]),
        (Q[19], Q[20]),
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
    elif name == "qw25q":
        return create_tii_qw25q(runcard)
    else:
        from qibolab.platforms.multiqubit import MultiqubitPlatform as Device

    return Device(name, runcard)
