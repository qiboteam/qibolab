import qibo
from qibo import gates
from qibo.backends import GlobalBackend
from qibo.models import Circuit

# define circuit
circuit = Circuit(1)
circuit.add(gates.U3(0, 0.1, 0.2, 0.3))
circuit.add(gates.M(0))

# set backend to qibolab
qibo.set_backend("qibolab", platform="dummy")

# disable the transpiler
GlobalBackend().transpiler = None

# execute circuit
result = circuit(nshots=1000)
