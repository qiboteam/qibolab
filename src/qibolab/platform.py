import itertools

import yaml
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
    channels |= ("readout", "drive")
    channels |= (f"flux-{i}" for i in range(6))

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
        platform.qubits[qubit].flux = channels[f"flux-{qubit}"]
        channels[f"flux-{qubit}"].qubit = platform.qubits[qubit]
        channels["readout"].attenuation = 0

    return platform


def create_tii_qw5q_gold(runcard, simulation_duration=None, address=None, cloud=False):
    """Create platform using Quantum Machines (QM) OPXs and Rohde Schwarz local oscillators.
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
    # readout
    channels |= ("L3-25_a", "L3-25_b")
    # feedback
    channels |= ("L2-5_a", "L2-5_b")
    # drive
    channels |= (f"L3-{i}" for i in range(11, 16))
    # flux
    channels |= (f"L4-{i}" for i in range(1, 6))
    # TWPA
    channels |= "L4-26"

    # Map controllers to qubit channels (HARDCODED)
    # readout
    channels["L3-25_a"].ports = [("con1", 10), ("con1", 9)]
    channels["L3-25_b"].ports = [("con2", 10), ("con2", 9)]
    # feedback
    channels["L2-5_a"].ports = [("con1", 2), ("con1", 1)]
    channels["L2-5_b"].ports = [("con2", 2), ("con2", 1)]
    # drive
    for i in range(1, 5):
        channels[f"L3-1{i}"].ports = [("con1", 2 * i), ("con1", 2 * i - 1)]
    channels["L3-15"].ports = [("con3", 2), ("con3", 1)]
    # flux
    for i in range(1, 6):
        channels[f"L4-{i}"].ports = [("con2", i)]

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
    local_oscillators = [
        LocalOscillator("lo_readout_a", "192.168.0.39"),
        LocalOscillator("lo_readout_b", "192.168.0.31"),
        LocalOscillator("lo_drive_low", "192.168.0.32"),
        LocalOscillator("lo_drive_mid", "192.168.0.33"),
        LocalOscillator("lo_drive_high", "192.168.0.34"),
        LocalOscillator("twpa_a", "192.168.0.35"),
    ]
    # Set LO parameters
    local_oscillators[0].frequency = 7_300_000_000
    local_oscillators[1].frequency = 7_900_000_000
    local_oscillators[2].frequency = 4_700_000_000
    local_oscillators[3].frequency = 5_600_000_000
    local_oscillators[4].frequency = 6_500_000_000
    local_oscillators[0].power = 18.0
    local_oscillators[1].power = 15.0
    for i in range(2, 5):
        local_oscillators[i].power = 16.0
    # Set TWPA parameters
    local_oscillators[5].frequency = 6_511_000_000
    local_oscillators[5].power = 4.5
    # Map LOs to channels
    channels["L3-25_a"].local_oscillator = local_oscillators[0]
    channels["L3-25_b"].local_oscillator = local_oscillators[1]
    channels["L3-15"].local_oscillator = local_oscillators[2]
    channels["L3-11"].local_oscillator = local_oscillators[2]
    channels["L3-12"].local_oscillator = local_oscillators[3]
    channels["L3-13"].local_oscillator = local_oscillators[4]
    channels["L3-14"].local_oscillator = local_oscillators[4]
    channels["L4-26"].local_oscillator = local_oscillators[5]

    instruments = [controller] + local_oscillators
    design = InstrumentDesign(instruments, channels)
    platform = DesignPlatform("qw5q_gold", design, runcard)

    # assign channels to qubits
    qubits = platform.qubits
    for q in [0, 1, 5]:
        qubits[q].readout = channels["L3-25_a"]
        qubits[q].feedback = channels["L2-5_a"]
    for q in [2, 3, 4]:
        qubits[q].readout = channels["L3-25_b"]
        qubits[q].feedback = channels["L2-5_b"]

    qubits[0].drive = channels["L3-15"]
    qubits[0].flux = channels["L4-5"]
    channels["L4-5"].qubit = qubits[0]
    for q in range(1, 5):
        qubits[q].drive = channels[f"L3-{10 + q}"]
        qubits[q].flux = channels[f"L4-{q}"]
        channels[f"L4-{q}"].qubit = qubits[q]
    # set maximum allowed bias values to protect amplifier
    # relevant only for qubits where an amplifier is used
    for q in range(5):
        platform.qubits[q].flux.max_bias = 0.2

    return platform


def create_tii_IQM5q(runcard, descriptor=None):
    """Create platform using Zurich Instrumetns (Zh) SHFQC, HDAWGs and PQSC.

    Instrument related parameters are hardcoded in ``__init__`` and ``setup``.

    Args:
        runcard (str): Path to the runcard file.
        descriptor (str): Instrument setup descriptor.
            If ``None`` it will attempt to connect to TII whole Zurich instruments setup.
    """
    # Create channel objects
    channels = ChannelMap()
    # readout
    channels |= "L3-31"
    # feedback
    channels |= "L2-7"
    # drive
    channels |= (f"L4-{i}" for i in range(15, 20))
    # flux qubits
    channels |= (f"L4-{i}" for i in range(6, 11))
    # flux couplers
    channels |= ChannelMap.from_names(*(f"L4-{i}" for i in range(11, 15)))

    # Map controllers to qubit channels
    # feedback
    channels["L3-31"].ports = [("device_shfqc", "[QACHANNELS/0/INPUT]")]
    channels["L3-31"].power_range = 10
    # readout
    channels["L2-7"].ports = [("device_shfqc", "[QACHANNELS/0/OUTPUT]")]
    channels["L2-7"].power_range = -25  # -5 for punchout
    # drive
    for i in range(5, 10):
        channels[f"L4-1{i}"].ports = [("device_shfqc", f"SGCHANNELS/{i-5}/OUTPUT")]
        channels[f"L4-1{i}"].power_range = -10

    # flux qubits (CAREFUL WITH THIS !!!)
    for i in range(6, 11):
        channels[f"L4-{i}"].ports = [("device_hdawg", f"SIGOUTS/{i-6}")]
        # channels[f"L4-{i}"].power_range = 0 #This may not be the default value find it

    # flux couplers (CAREFUL WITH THIS !!!)
    for i in range(11, 14):
        channels[f"L4-{i}"].ports = [("device_hdawg", f"SIGOUTS/{i-11+5}")]
        # channels[f"L4-{i}"].power_range = 0 #This may not be the default value find it

    channels[f"L4-14"].ports = [("device_hdawg2", f"SIGOUTS/0")]
    # channels["L4-14"].power_range = 0 #This may not be the default value find it

    # Instantiate Zh set of instruments[They work as one]
    from qibolab.instruments.dummy_oscillator import (
        DummyLocalOscillator as LocalOscillator,
    )
    from qibolab.instruments.zhinst import Zurich

    if descriptor is None:
        instruments = {
            "SHFQC": [{"address": "DEV12146", "uid": "device_shfqc"}],
            "HDAWG": [{"address": "DEV8660", "uid": "device_hdawg"}, {"address": "DEV8673", "uid": "device_hdawg2"}],
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

        descriptor_yaml = yaml.dump(descriptor, sort_keys=False)

    controller = Zurich("EL_ZURO", descriptor_yaml, use_emulation=False)

    # set time of flight for readout integration (HARDCODED)
    controller.time_of_flight = 280
    controller.smearing = 100

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

    instruments = [controller] + local_oscillators
    design = InstrumentDesign(instruments, channels)
    platform = DesignPlatform("IQM5q", design, runcard)
    platform.resonator_type = "2D"

    # assign channels to qubits and sweetspots(operating points)
    qubits = platform.qubits
    for q in range(0, 5):
        qubits[q].feedback = channels["L3-31"]
        qubits[q].readout = channels["L2-7"]

    for q in range(0, 5):
        qubits[q].drive = channels[f"L4-{15 + q}"]
        qubits[q].flux = channels[f"L4-{6 + q}"]
        channels[f"L4-{6 + q}"].qubit = qubits[q]

    # assign channels to couplers and sweetspots(operating points)
    for c in range(0, 2):
        qubits[f"c{c}"].flux = channels[f"L4-{11 + c}"]
        channels[f"L4-{11 + c}"].qubit = qubits[f"c{c}"]
    for c in range(3, 5):
        qubits[f"c{c}"].flux = channels[f"L4-{10 + c}"]
        channels[f"L4-{10 + c}"].qubit = qubits[f"c{c}"]

    # assign qubits to couplers
    for c in itertools.chain(range(0, 2), range(3, 5)):
        qubits[f"c{c}"].flux_coupler = [qubits[c]]
        qubits[f"c{c}"].flux_coupler.append(qubits[2])

    return platform


def create_tii_rfsoc4x2(runcard, address=None):
    """Create platform using QICK project on the RFSoC4x2 board
    IPs and other instrument related parameters are hardcoded in ``__init__`` and ``setup``.
    Args:
        runcard (str): Path to the runcard file.
        address (str): Address and port for the QICK board.
            If ``None`` it will attempt to connect to TII instruments.
    """
    from qibolab.instruments.rfsoc import TII_RFSOC4x2
    from qibolab.instruments.rohde_schwarz import SGS100A as LocalOscillator

    # Create channel objects
    channels = ChannelMap()
    channels |= ChannelMap.from_names("L3-18_ro")  # readout (DAC)
    channels |= ChannelMap.from_names("L2-RO")  # feedback (readout DAC)
    channels |= ChannelMap.from_names("L3-18_qd")  # drive

    # Map controllers to qubit channels (HARDCODED)
    channels["L3-18_ro"].ports = [("o0", 0)]  # readout
    channels["L2-RO"].ports = [("i0", 0)]  # feedback
    channels["L3-18_qd"].ports = [("o1", 1)]  # drive

    local_oscillators = [
        LocalOscillator("twpa_a", "192.168.0.32"),
    ]
    local_oscillators[0].frequency = 6_200_000_000
    local_oscillators[0].power = -1

    # Instantiate QICK instruments
    if address is None:
        address = "192.168.0.72:6000"
    controller = TII_RFSOC4x2("tii_rfsoc4x2", address)
    design = InstrumentDesign([controller] + local_oscillators, channels)

    platform = DesignPlatform("tii_rfsoc4x2", design, runcard)

    # assign channels to qubits
    qubits = platform.qubits
    qubits[0].readout = channels["L3-18_ro"]
    qubits[0].feedback = channels["L2-RO"]
    qubits[0].drive = channels["L3-18_qd"]  # Create channel objects

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

    if name == "tii1q" or name == "tii5q" or name == "qili" or name == "zhinst":
        from qibolab.platforms.multiqubit import MultiqubitPlatform as Device
    if name == "dummy":
        return create_dummy(runcard)
    elif name == "icarusq":
        from qibolab.platforms.icplatform import ICPlatform as Device
    elif name == "qw5q_gold":
        return create_tii_qw5q_gold(runcard)
    elif name == "iqm5q":
        return create_tii_IQM5q(runcard)
    elif name == "tii1q_b1":
        return create_tii_rfsoc4x2(runcard)
    else:
        from qibolab.platforms.multiqubit import MultiqubitPlatform as Device

    return Device(name, runcard)
