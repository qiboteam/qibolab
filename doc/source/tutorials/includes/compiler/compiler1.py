from qibo import gates
from qibo.models import Circuit

from qibolab.backends import QibolabBackend
from qibolab.pulses import PulseSequence

# define the circuit
circuit = Circuit(1)
circuit.add(gates.X(0))
circuit.add(gates.M(0))


# define a compiler rule that translates X to the pi-pulse
def x_rule(gate, platform):
    """X gate applied with a single pi-pulse."""
    qubit = gate.target_qubits[0]
    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit, start=0))
    return sequence, {}


# the empty dictionary is needed because the X gate does not require any virtual Z-phases

backend = QibolabBackend(platform="dummy")
# disable the transpiler
backend.transpiler = None
# register the new X rule in the compiler
backend.compiler[gates.X] = x_rule

# execute the circuit
result = backend.execute_circuit(circuit, nshots=1000)
