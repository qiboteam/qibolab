import networkx as nx
from qibo import gates
from qibo.models import Circuit

from qibolab.transpilers.placer import Trivial
from qibolab.transpilers.router import Sabre


def star_connectivity():
    Q = [i for i in range(5)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [(Q[i], Q[2]) for i in range(5) if i != 2]
    chip.add_edges_from(graph_list)
    return chip


def matched_circuit():
    """Return a simple circuit that can be executed on star connectivity"""
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.Z(1))
    circuit.add(gates.CZ(2, 0))
    circuit.add(gates.CZ(4, 3))
    return circuit


# TODO: fix circuit dag
placer = Trivial()
layout_circ = Circuit(5)
initial_layout = placer(layout_circ)
router = Sabre(connectivity=star_connectivity())
routed_circuit = router(circuit=matched_circuit(), initial_layout=initial_layout)
print(router.added_swaps())
print(routed_circuit.draw())
