import networkx as nx
import numpy as np
from qibo import Circuit, gates

from qibolab.transpilers.pipeline import assert_circuit_equivalence
from qibolab.transpilers.router import Sabre

circ = Circuit(3)
circ.add(gates.X(0))
circ.add(gates.Z(1))
circ.add(gates.CZ(1, 0))
print("Original Circuit")
print(circ.draw())
print(circ())
print(circ(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
print(circ(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])))

connectivity = nx.Graph()
connectivity.add_nodes_from([0, 1, 2])
connectivity.add_edges_from([(0, 1), (1, 2)])
layout = {"q0": 0, "q1": 2, "q2": 1}
# layout = {"q0": 0, "q1": 1, "q2":2}
router = Sabre(connectivity=connectivity, lookahead=0)
transpiled, final_map = router(circuit=circ, initial_layout=layout)
print("Transpiled")
print(transpiled.draw())
print(transpiled())
print(transpiled(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
print(transpiled(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])))
for physical, logical in final_map.items():
    print("{}: {}".format(physical, logical))
assert_circuit_equivalence(original_circuit=circ, transpiled_circuit=transpiled, final_map=final_map)
