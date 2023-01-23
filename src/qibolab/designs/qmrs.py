from qibo.config import log, raise_error

from qibolab.designs.abstract import AbstractInstrumentDesign
from qibolab.platforms.utils import Channel


class QMRSDesign(AbstractInstrumentDesign):
    """Instrument design for Quantum Machines (QM) OPXs and Rohde Schwarz local oscillators.

    IPs and other instrument related parameters are hardcoded in ``__init__`` and ``setup``.

    Args:
        address (str): Address and port for the QM OPX cluster.
            Default is the current address for the instruments in TII lab.
        simulation_duration (int): Duration for the simulation in ns.
            If given the compiler simulator will be used instead of the actual hardware.
            Default is ``None`` which falls back to the hardware.
        cloud (bool): See :class:`qibolab.instruments.qmsim.QMSim` for details.
            Relevant only when ``simulation_duration`` is given.

    Attributes:
        is_connected (bool): Boolean that shows whether instruments are connected.
        opx (:class:`qibolab.instruments.qm.QMOPX`): Object used for controlling the QM OPXs.
        local_oscillators (list): List of local oscillator objects.
            instrument objects.
    """

    def __init__(self, address="192.168.0.1:80", simulation_duration=None, cloud=False):
        from qibolab.instruments.rohde_schwarz import SGS100A

        self.is_connected = False

        # Instantiate QM OPX instruments
        if simulation_duration is None:
            from qibolab.instruments.qm import QMOPX

            self.opx = QMOPX("qmopx", address)
        else:
            from qibolab.instruments.qmsim import QMSim

            self.opx = QMSim("qmopx", address, simulation_duration, cloud)

        # Map controllers to qubit channels (HARDCODED)
        # readout
        Channel("L3-25_a").ports = [
            ("con1", 10),
            ("con1", 9),
        ]
        Channel("L3-25_b").ports = [
            ("con2", 10),
            ("con2", 9),
        ]
        # feedback
        Channel("L2-5").ports = [("con1", 2), ("con1", 1)]
        # drive
        Channel("L3-11").ports = [("con1", 2), ("con1", 1)]
        Channel("L3-12").ports = [("con1", 4), ("con1", 3)]
        Channel("L3-13").ports = [("con1", 6), ("con1", 5)]
        Channel("L3-14").ports = [("con1", 8), ("con1", 7)]
        Channel("L3-15").ports = [("con3", 2), ("con3", 1)]
        # flux
        Channel("L4-1").ports = [("con2", 1)]
        Channel("L4-2").ports = [("con2", 2)]
        Channel("L4-3").ports = [("con2", 3)]
        Channel("L4-4").ports = [("con2", 4)]
        Channel("L4-5").ports = [("con2", 5)]

        # Instantiate local oscillators (HARDCODED)
        self.local_oscillators = [
            SGS100A("lo_readout_a", "192.168.0.39"),
            SGS100A("lo_readout_b", "192.168.0.31"),
            SGS100A("lo_drive_low", "192.168.0.32"),
            SGS100A("lo_drive_mid", "192.168.0.33"),
            SGS100A("lo_drive_high", "192.168.0.34"),
        ]

        # Map LOs to channels
        Channel("L3-25_a").local_oscillator = self.local_oscillators[0]
        Channel("L3-25_b").local_oscillator = self.local_oscillators[1]
        Channel("L3-15").local_oscillator = self.local_oscillators[2]
        Channel("L3-11").local_oscillator = self.local_oscillators[2]
        Channel("L3-12").local_oscillator = self.local_oscillators[3]
        Channel("L3-13").local_oscillator = self.local_oscillators[4]
        Channel("L3-14").local_oscillator = self.local_oscillators[4]

        # Set default LO parameters in the channel
        Channel("L3-25_a").lo_frequency = 7_300_000_000
        Channel("L3-25_b").lo_frequency = 7_900_000_000
        Channel("L3-15").lo_frequency = 4_700_000_000
        Channel("L3-11").lo_frequency = 4_700_000_000
        Channel("L3-12").lo_frequency = 5_600_000_000
        Channel("L3-13").lo_frequency = 6_500_000_000
        Channel("L3-14").lo_frequency = 6_500_000_000

        Channel("L3-25_a").lo_power = 18.0
        Channel("L3-25_b").lo_power = 15.0
        Channel("L3-15").lo_power = 16.0
        Channel("L3-11").lo_power = 16.0
        Channel("L3-12").lo_power = 16.0
        Channel("L3-13").lo_power = 16.0
        Channel("L3-14").lo_power = 16.0

        # drop the local oscillators to avoid connecting when simulation is used
        if simulation_duration is not None:
            self.local_oscillators = []

    def connect(self):
        self.opx.connect()
        if not self.is_connected:
            for lo in self.local_oscillators:
                try:
                    log.info(f"Connecting to instrument {lo}.")
                    lo.connect()
                except Exception as exception:
                    raise_error(
                        RuntimeError,
                        f"Cannot establish connection to {lo} instruments. Error captured: '{exception}'",
                    )
            self.is_connected = True

    def setup(self, qubits, **kwargs):
        relaxation_time = kwargs["relaxation_time"]
        time_of_flight = kwargs["time_of_flight"]
        smearing = kwargs["smearing"]

        for qubit in qubits:
            if qubit.flux is not None:
                # set flux offset
                qubit.flux.offset = qubit.characterization.sweetspot
                # Set flux filters (useful for CZ gates)
                qubit.flux.filter = {
                    "feedforward": qubit.characterization.ff_filter,
                    "feedback": qubit.characterization.fb_filter,
                }
            # set LO frequencies
            for channel in [qubit.readout, qubit.drive]:
                if channel is not None and channel.local_oscillator is not None:
                    # set LO frequency
                    lo = channel.local_oscillator
                    frequency = channel.lo_frequency
                    if lo.is_connected:
                        lo.setup(frequency=frequency, power=channel.lo_power)
                    else:
                        log.warn(f"There is no connection to {lo}. Frequencies were not set.")

        # setup QM
        self.opx.setup(qubits, relaxation_time, time_of_flight, smearing)

    def start(self):
        self.opx.start()
        if self.is_connected:
            for lo in self.local_oscillators:
                lo.start()

    def stop(self):
        if self.is_connected:
            for lo in self.local_oscillators:
                lo.stop()
        self.opx.stop()

    def disconnect(self):
        if self.is_connected:
            self.opx.disconnect()
            for lo in self.local_oscillators:
                lo.disconnect()
            self.is_connected = False

    def sweep(self, qubits, sequence, *sweepers, nshots=1024, average=True):
        return self.opx.sweep(qubits, sequence, *sweepers, nshots=nshots, average=average)

    def play(self, qubits, sequence, nshots=1024):
        return self.opx.play(qubits, sequence, nshots)
