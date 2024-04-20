from qibolab.native import (
    NativePulse,
    NativeSequence,
    SingleQubitNatives,
    TwoQubitNatives,
)
from qibolab.pulses import PulseType
from qibolab.qubits import Qubit, QubitPair

# create the qubit objects
qubit0 = Qubit(0)
qubit1 = Qubit(1)

# assign single-qubit native gates to each qubit
qubit0.native_gates = SingleQubitNatives(
    RX=NativePulse(
        name="RX",
        duration=40,
        amplitude=0.05,
        shape="Gaussian(5)",
        pulse_type=PulseType.DRIVE,
        qubit=qubit0,
        frequency=int(4.7e9),
    ),
    MZ=NativePulse(
        name="MZ",
        duration=1000,
        amplitude=0.005,
        shape="Rectangular()",
        pulse_type=PulseType.READOUT,
        qubit=qubit0,
        frequency=int(7e9),
    ),
)
qubit1.native_gates = SingleQubitNatives(
    RX=NativePulse(
        name="RX",
        duration=40,
        amplitude=0.05,
        shape="Gaussian(5)",
        pulse_type=PulseType.DRIVE,
        qubit=qubit1,
        frequency=int(5.1e9),
    ),
    MZ=NativePulse(
        name="MZ",
        duration=1000,
        amplitude=0.005,
        shape="Rectangular()",
        pulse_type=PulseType.READOUT,
        qubit=qubit1,
        frequency=int(7.5e9),
    ),
)

# define the pair of qubits
pair = QubitPair(qubit0, qubit1)
pair.native_gates = TwoQubitNatives(
    CZ=NativeSequence(
        name="CZ",
        pulses=[
            NativePulse(
                name="CZ1",
                duration=30,
                amplitude=0.005,
                shape="Rectangular()",
                pulse_type=PulseType.FLUX,
                qubit=qubit1,
            )
        ],
    )
)
