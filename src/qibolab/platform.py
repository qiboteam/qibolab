from qibo.config import raise_error

from qibolab.designs.channels import Channel, ChannelMap
from qibolab.designs.mixer import MixerInstrumentDesign
from qibolab.platforms.platform import DesignPlatform


def create_tii_rfsocZCU111(runcard, address=None):
    """Create platform using QICK project on the ZCU111 board and EraSynth local oscillator for the Readout
    IPs and other instrument related parameters are hardcoded in ``__init__`` and ``setup``.
    Args:
        runcard (str): Path to the runcard file.
        address (str): Address and port for the QICK board.
            If ``None`` it will attempt to connect to TII instruments.
    """
    from qibolab.instruments.dummy_oscillator import (
        DummyLocalOscillator as LocalOscillator,
    )
    from qibolab.instruments.rfsoc import TII_RFSOC_ZCU111

    # Create channel objects
    channels = ChannelMap()

    # QUBIT 0
    # readout
    channels |= ChannelMap.from_names("L3-18_ro")
    # feedback
    channels |= ChannelMap.from_names("L2_3-RO")
    # drive
    channels |= ChannelMap.from_names("L4-23_qd")
    # Flux
    channels |= ChannelMap.from_names("L1-16_fl")
    # QUBIT 1
    # readout
    channels |= ChannelMap.from_names("L3-18_ro")
    # feedback
    channels |= ChannelMap.from_names("L2-3-RO")
    # drive
    channels |= ChannelMap.from_names("L4-24_qd")
    # Flux
    channels |= ChannelMap.from_names("L1-17_fl")
    # QUBIT 2
    # readout
    channels |= ChannelMap.from_names("L3-18_ro")
    # feedback
    channels |= ChannelMap.from_names("L2-3-RO")
    # drive
    channels |= ChannelMap.from_names("L4-25_qd")
    # Flux
    channels |= ChannelMap.from_names("L1-18_fl")

    # Map controllers to qubit channels (HARDCODED)
    # Qubit 0
    # readout
    channels["L3-18_ro"].ports = [("dac6", 6)]
    # feedback
    channels["L2-3-RO"].ports = [("adc0", 0)]
    # drive
    channels["L4-23_qd"].ports = [("dac0", 0)]
    # Flux
    channels["L1-16_fl"].ports = [("dac3", 3)]
    # Qubit 1
    # Readout
    channels["L3-18_ro"].ports = [("dac3", 3)]
    # feedback
    channels["L2-3-RO"].ports = [("adc0", 1)]
    # drive
    channels["L4-24_qd"].ports = [("dac1", 1)]
    # Flux
    channels["L1-17_fl"].ports = [("dac4", 4)]
    # Qubit 2
    # Readout
    channels["L3-18_ro"].ports = [("dac6", 6)]
    # feedback
    channels["L2-3-RO"].ports = [("adc0", 2)]
    # drive
    channels["L4-25_qd"].ports = [("dac2", 2)]
    # Flux
    channels["L1-18_fl"].ports = [("dac5", 5)]

    # Instantiate QICK instruments

    if address is None:
        # connect to TII instruments for simulation
        address = "192.168.2.81:6000"

    controller = TII_RFSOC_ZCU111("tii_rfsocZCU111", address)

    # Instantiate local oscillators (HARDCODED) # TODO local oscillators should not be needed
    local_oscillators = [
        LocalOscillator("twpa", "192.168.0.35"),
    ]
    # Set TWPA parameters
    # local_oscillators[0].frequency = 6_511_000_000
    # local_oscillators[0].power = 4.5

    # Map LOs to channels
    # channels["L4-26"].local_oscillator = local_oscillators[0]  # TODO find the real channel

    design = MixerInstrumentDesign(controller, channels, local_oscillators)
    platform = DesignPlatform("tii_rfsocZCU111", design, runcard)

    # assign channels to qubits
    qubits = platform.qubits
    qubits[0].readout = channels["L3-18_ro"]
    qubits[0].feedback = channels["L2-3-RO"]
    qubits[0].drive = channels["L4-23_qd"]
    qubits[1].readout = channels["L3-18_ro"]
    qubits[1].feedback = channels["L2-3-RO"]
    qubits[1].drive = channels["L4-24_qd"]
    qubits[2].readout = channels["L3-18_ro"]
    qubits[2].feedback = channels["L2-3-RO"]
    qubits[2].drive = channels["L4-25_qd"]
    return platform


def create_tii_rfsoc4x2(runcard: str):
    """Create platform using QICK project on the RFSoS4x2 board and Rohde Schwarz local oscillator for the TWPA
    IPs and other instrument related parameters are hardcoded in ``__init__`` and ``setup``.
    Args:
        runcard (str): Path to the runcard file.
    """
    from qibolab.instruments.rfsoc import TII_RFSOC4x2

    # Create channel objects
    channels = ChannelMap()
    # TODO: channels names are not correct
    channels |= ChannelMap.from_names("L3-18_ro")  # readout (ADC)
    channels |= ChannelMap.from_names("L2-RO")  # feedback (readout DAC)
    channels |= ChannelMap.from_names("L3-18_qd")  # drive
    channels |= ChannelMap.from_names("L4-26")  # TWPA

    # Map controllers to qubit channels (HARDCODED)
    channels["L3-18_ro"].ports = [("o0", 0)]  # readout
    channels["L2-RO"].ports = [("i0", 0)]  # feedback
    channels["L3-18_qd"].ports = [("o1", 1)]  # drive

    # Instantiate QICK instruments
    controller = TII_RFSOC4x2("tii_rfsoc4x2")
    design = MixerInstrumentDesign(controller, channels)  # TODO: use single instrument design

    platform = DesignPlatform("tii_rfsoc4x2", design, runcard)

    # assign channels to qubits
    qubits = platform.qubits
    qubits[0].readout = channels["L3-18_ro"]
    qubits[0].feedback = channels["L2-RO"]
    qubits[0].drive = channels["L3-18_qd"]

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
    channels |= ChannelMap.from_names("L3-25_a", "L3-25_b")
    # feedback
    channels |= ChannelMap.from_names("L2-5_a", "L2-5_b")
    # drive
    channels |= ChannelMap.from_names(*(f"L3-{i}" for i in range(11, 16)))
    # flux
    channels |= ChannelMap.from_names(*(f"L4-{i}" for i in range(1, 6)))
    # TWPA
    channels |= ChannelMap.from_names("L4-26")

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

    design = MixerInstrumentDesign(controller, channels, local_oscillators)
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
        from qibolab.platforms.dummy import DummyPlatform as Device
    elif name == "icarusq":
        from qibolab.platforms.icplatform import ICPlatform as Device
    elif name == "qw5q_gold":
        return create_tii_qw5q_gold(runcard)
    elif name == "rfsoc":
        return create_tii_rfsoc4x2(runcard)
    elif name == "tii_rfsocZCU111":
        return create_tii_rfsocZCU111(runcard)
    else:
        from qibolab.platforms.multiqubit import MultiqubitPlatform as Device

    return Device(name, runcard)
