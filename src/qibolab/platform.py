from qibo.config import raise_error

from qibolab.designs.channels import Channel, ChannelMap
from qibolab.designs.mixer import MixerInstrumentDesign
from qibolab.platforms.platform import DesignPlatform


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


# TODO: Treat couplers as qubits but without readout
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
    channels |= ChannelMap.from_names("L3-31")
    # feedback
    channels |= ChannelMap.from_names("L1-2")
    # drive
    channels |= ChannelMap.from_names(*(f"L4-{i}" for i in range(15, 20)))
    # flux qubits
    channels |= ChannelMap.from_names(*(f"L4-{i}" for i in range(6, 11)))
    # flux couplers
    channels |= ChannelMap.from_names(*(f"L4-{i}" for i in range(11, 15)))
    # TWPA
    # channels |= ChannelMap.from_names("L3-10")

    # Map controllers to qubit channels (HARDCODED)
    # readout
    channels["L3-31"].ports = [("device_shfqc", "[QACHANNELS/0/INPUT]")]
    channels["L3-31"].power_range = -5
    # feedback
    channels["L1-2"].ports = [("device_shfqc", "[QACHANNELS/0/OUTPUT]")]
    channels["L1-2"].power_range = 10
    # drive
    for i in range(5, 10):
        channels[f"L4-1{i}"].ports = [("device_shfqc", f"SGCHANNELS/{i-5}/OUTPUT")]
        channels[f"L4-1{i}"].power_range = 5
    # flux qubits
    for i in range(6, 11):
        channels[f"L4-{i}"].ports = [("device_hdawg", f"SIGOUTS/{i-6}")]
        channels[f"L4-{i}"].offset = 0.1
    # flux couplers
    for i in range(11, 15):
        channels[f"L4-{i}"].ports = [("device_hdawg", f"SIGOUTS/{i-11+5}")]
        channels[f"L4-{i}"].offset = 0.1

    # DEVICE HDWAG1 and HDAWG2 ???

    # Instantiate Zh set of instruments[They work as one]
    from qibolab.instruments.dummy_oscillator import (
        DummyLocalOscillator as LocalOscillator,
    )
    from qibolab.instruments.rohde_schwarz import SGS100A as TWPA_Oscillator
    from qibolab.instruments.zhinst import Zurich

    if descriptor is None:
        descriptor = """\
        instruments:
            SHFQC:
            - address: DEV12146
              uid: device_shfqc
            HDAWG:
            - address: DEV8660
              uid: device_hdawg
            PQSC:
            - address: DEV10055
              uid: device_pqsc


        connections:
            device_shfqc:
                - iq_signal: q0/drive_line
                  ports: SGCHANNELS/0/OUTPUT
                - iq_signal: q1/drive_line
                  ports: SGCHANNELS/1/OUTPUT
                - iq_signal: q2/drive_line
                  ports: SGCHANNELS/2/OUTPUT
                - iq_signal: q3/drive_line
                  ports: SGCHANNELS/3/OUTPUT
                - iq_signal: q4/drive_line
                  ports: SGCHANNELS/4/OUTPUT
                - iq_signal: q0/measure_line
                  ports: [QACHANNELS/0/OUTPUT]
                - acquire_signal: q0/acquire_line
                  ports: [QACHANNELS/0/INPUT]
                - iq_signal: q1/measure_line
                  ports: [QACHANNELS/0/OUTPUT]
                - acquire_signal: q1/acquire_line
                  ports: [QACHANNELS/0/INPUT]
                - iq_signal: q2/measure_line
                  ports: [QACHANNELS/0/OUTPUT]
                - acquire_signal: q2/acquire_line
                  ports: [QACHANNELS/0/INPUT]
                - iq_signal: q3/measure_line
                  ports: [QACHANNELS/0/OUTPUT]
                - acquire_signal: q3/acquire_line
                  ports: [QACHANNELS/0/INPUT]
                - iq_signal: q4/measure_line
                  ports: [QACHANNELS/0/OUTPUT]
                - acquire_signal: q4/acquire_line
                  ports: [QACHANNELS/0/INPUT]

            device_hdawg:
                - rf_signal: q0/flux_line
                  ports: SIGOUTS/0
                - rf_signal: q1/flux_line
                  ports: SIGOUTS/1
                - rf_signal: q2/flux_line
                  ports: SIGOUTS/2
                - rf_signal: q3/flux_line
                  ports: SIGOUTS/3
                - rf_signal: q4/flux_line
                  ports: SIGOUTS/4
                - rf_signal: q03/flux_line
                  ports: SIGOUTS/4
                - rf_signal: q13/flux_line
                  ports: SIGOUTS/4
                - rf_signal: q23/flux_line
                  ports: SIGOUTS/4

            device_pqsc:
                - internal_clock_signal
                - to: device_hdawg
                  port: ZSYNCS/2
                - to: device_shfqc
                  port: ZSYNCS/0
        """

    controller = Zurich("EL_ZURO", descriptor, use_emulation=False)

    # Instantiate local oscillators (HARDCODED)
    local_oscillators = [
        LocalOscillator("lo_readout", None),
        LocalOscillator("lo_drive_0", None),
        LocalOscillator("lo_drive_1", None),
        LocalOscillator("lo_drive_2", None),
        LocalOscillator("lo_drive_3", None),
        LocalOscillator("lo_drive_4", None),
        LocalOscillator("lo_drive_5", None),
        # TWPA_Oscillator("TWPA", "192.168.0.35"),
    ]
    # Set Dummy LO parameters
    local_oscillators[0].frequency = 7_300_000_000
    local_oscillators[1].frequency = 7_900_000_000
    local_oscillators[2].frequency = 7_900_000_000
    local_oscillators[3].frequency = 5_600_000_000
    local_oscillators[4].frequency = 5_600_000_000
    local_oscillators[5].frequency = 5_800_000_000
    local_oscillators[6].frequency = 5_800_000_000

    # Set TWPA pump LO parameters
    # local_oscillators[7].frequency = 6_511_000_000
    # local_oscillators[7].power = 4.5

    # Map LOs to channels
    channels["L3-31"].local_oscillator = local_oscillators[0]
    channels["L4-15"].local_oscillator = local_oscillators[1]
    channels["L4-16"].local_oscillator = local_oscillators[2]
    channels["L4-17"].local_oscillator = local_oscillators[3]
    channels["L4-18"].local_oscillator = local_oscillators[4]
    channels["L4-19"].local_oscillator = local_oscillators[5]
    # channels["Witness???"].local_oscillator = local_oscillators[6]
    # channels["L3-10"].local_oscillator = local_oscillators[7]

    design = MixerInstrumentDesign(controller, channels, local_oscillators)
    platform = DesignPlatform("IQM5q", design, runcard)

    # assign channels to qubits
    qubits = platform.qubits
    for q in range(0, 5):
        qubits[q].readout = channels["L3-31"]
        qubits[q].feedback = channels["L1-2"]

    for q in range(0, 5):
        qubits[q].drive = channels[f"L4-{15 + q}"]
        qubits[q].flux = channels[f"L4-{6 + q}"]
        # channels[f"L4-{6 + q}"].qubit = qubits[q]

    # assign channels to couplers
    couplers = platform.couplers

    for c in range(0, 2):
        couplers[c].flux_coupler = channels[f"L4-{11 + c}"]

    for c in range(3, 5):
        couplers[c].flux_coupler = channels[f"L4-{11 + c}"]

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
    else:
        from qibolab.platforms.multiqubit import MultiqubitPlatform as Device

    return Device(name, runcard)
