from qibo.config import raise_error

from qibolab.designs.basic import BasicInstrumentDesign
from qibolab.designs.mixed import MixedInstrumentDesign
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
    # Instantiate QM OPX instruments
    if simulation_duration is None:
        from qibolab.instruments.qm import QMOPX
        from qibolab.instruments.rohde_schwarz import SGS100A

        controller = QMOPX("qmopx", "192.168.0.1:80")

        # Instantiate local oscillators (HARDCODED)
        local_oscillators = [
            SGS100A("lo_readout_a", "192.168.0.39"),
            SGS100A("lo_readout_b", "192.168.0.31"),
            SGS100A("lo_drive_low", "192.168.0.32"),
            SGS100A("lo_drive_mid", "192.168.0.33"),
            SGS100A("lo_drive_high", "192.168.0.34"),
            SGS100A("twpa_a", "192.168.0.35"),
        ]
        design = MixedInstrumentDesign(controller, local_oscillators)

    else:
        from qibolab.instruments.qmsim import QMSim

        if address is None:
            # connect to TII instruments for simulation
            address = "192.168.0.1:80"

        controller = QMSim("qmopx", address, simulation_duration, cloud)
        # avoid connecting to local oscillators when simulation is used
        local_oscillators = []
        design = BasicInstrumentDesign(controller)

    platform = DesignPlatform("qw5q_gold", design, runcard)
    # Map controllers to qubit channels (HARDCODED)
    channels = platform.channels
    # readout
    channels["L3-25_a"].ports = [("con1", 10), ("con1", 9)]
    channels["L3-25_b"].ports = [("con2", 10), ("con2", 9)]
    # feedback
    channels["L2-5"].ports = [("con1", 2), ("con1", 1)]
    # drive
    channels["L3-11"].ports = [("con1", 2), ("con1", 1)]
    channels["L3-12"].ports = [("con1", 4), ("con1", 3)]
    channels["L3-13"].ports = [("con1", 6), ("con1", 5)]
    channels["L3-14"].ports = [("con1", 8), ("con1", 7)]
    channels["L3-15"].ports = [("con3", 2), ("con3", 1)]
    # flux
    channels["L4-1"].ports = [("con2", 1)]
    channels["L4-2"].ports = [("con2", 2)]
    channels["L4-3"].ports = [("con2", 3)]
    channels["L4-4"].ports = [("con2", 4)]
    channels["L4-5"].ports = [("con2", 5)]

    # Map LOs to channels
    if local_oscillators:
        channels["L3-25_a"].local_oscillator = local_oscillators[0]
        channels["L3-25_b"].local_oscillator = local_oscillators[1]
        channels["L3-15"].local_oscillator = local_oscillators[2]
        channels["L3-11"].local_oscillator = local_oscillators[2]
        channels["L3-12"].local_oscillator = local_oscillators[3]
        channels["L3-13"].local_oscillator = local_oscillators[4]
        channels["L3-14"].local_oscillator = local_oscillators[4]
        channels["L4-26"].local_oscillator = local_oscillators[5]

    # Set default LO parameters in the channel
    channels["L3-25_a"].lo_frequency = 7_300_000_000
    channels["L3-25_b"].lo_frequency = 7_900_000_000
    channels["L3-15"].lo_frequency = 4_700_000_000
    channels["L3-11"].lo_frequency = 4_700_000_000
    channels["L3-12"].lo_frequency = 5_600_000_000
    channels["L3-13"].lo_frequency = 6_500_000_000
    channels["L3-14"].lo_frequency = 6_500_000_000
    channels["L3-25_a"].lo_power = 18.0
    channels["L3-25_b"].lo_power = 15.0
    channels["L3-15"].lo_power = 16.0
    channels["L3-11"].lo_power = 16.0
    channels["L3-12"].lo_power = 16.0
    channels["L3-13"].lo_power = 16.0
    channels["L3-14"].lo_power = 16.0
    # Map TWPA to channels
    channels["L4-26"].lo_frequency = 6_558_000_000
    channels["L4-26"].lo_power = 2.5

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
