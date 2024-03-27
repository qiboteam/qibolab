from qibolab.couplers import Coupler
from qibolab.qubits import Qubit, QubitPair
from qibolab.pulses import PulseType
from qibolab.native import (
    NativePulse,
    NativeSequence,
    SingleQubitNatives,
    TwoQubitNatives,
)

# create the qubit and coupler objects
qubit0 = Qubit(0)
qubit1 = Qubit(1)
coupler_01 = Coupler(0)

# assign single-qubit native gates to each qubit
# Look above example

# define the pair of qubits
pair = QubitPair(qubit0, qubit1, coupler_01)
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