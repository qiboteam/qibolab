from qibolab import Platform
from qibolab.qubits import Qubit
from qibolab.pulses import PulseType
from qibolab.channels import ChannelMap, Channel
from qibolab.native import NativePulse, SingleQubitNatives
from qibolab.instruments.dummy import DummyInstrument


def create():
    # Create a controller instrument
    instrument = DummyInstrument("my_instrument", "0.0.0.0:0")

    # Create channel objects and assign to them the controller ports
    channels = ChannelMap()
    channels |= Channel("ch1out", port=instrument["o1"])
    channels |= Channel("ch2", port=instrument["o2"])
    channels |= Channel("ch1in", port=instrument["i1"])

    # create the qubit object
    qubit = Qubit(0)

    # assign native gates to the qubit
    qubit.native_gates = SingleQubitNatives(
        RX=NativePulse(
            name="RX",
            duration=40,
            amplitude=0.05,
            shape="Gaussian(5)",
            pulse_type=PulseType.DRIVE,
            qubit=qubit,
            frequency=int(4.5e9),
        ),
        MZ=NativePulse(
            name="MZ",
            duration=1000,
            amplitude=0.005,
            shape="Rectangular()",
            pulse_type=PulseType.READOUT,
            qubit=qubit,
            frequency=int(7e9),
        ),
    )

    # assign channels to the qubit
    qubit.readout = channels["ch1out"]
    qubit.feedback = channels["ch1in"]
    qubit.drive = channels["ch2"]

    # create dictionaries of the different objects
    qubits = {qubit.name: qubit}
    pairs = {}  # empty as for single qubit we have no qubit pairs
    instruments = {instrument.name: instrument}

    # allocate and return Platform object
    return Platform("my_platform", qubits, pairs, instruments, resonator_type="3D")
