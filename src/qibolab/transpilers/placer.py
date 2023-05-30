import random

import networkx as nx
from qibo.config import log, raise_error
from qibo.models import Circuit

from qibolab.transpilers.abstract import Placer, create_circuit_repr


def check_placement(circuit: Circuit, layout: dict, verbose=False) -> bool:
    if not check_mapping_consistency(layout, verbose=verbose):
        return False
    if circuit.nqubits == len(layout):
        if verbose:
            log.info("Layout can be used on circuit.")
        return True
    elif circuit.nqubits > len(layout):
        if verbose:
            log.info("Layout can't be used on circuit. The circuit requires more qubits.")
        return False
    else:
        if verbose:
            log.info("Layout can't be used on circuit. Ancillary extra qubits need to be added to the circuit.")
        return False


def check_mapping_consistency(layout, verbose=False):
    values = list(layout.values())
    values.sort()
    keys = list(layout.keys())
    ref_keys = list("q" + str(i) for i in range(len(keys)))
    if keys != ref_keys:
        if verbose:
            log.info("Some physical qubits in the layout may be missing or duplicated")
        return False
    if values != list(range(len(values))):
        if verbose:
            log.info("Some logical qubits in the layout may be missing or duplicated")
        return False
    return True


class Trivial(Placer):
    """Place qubits trivially, same logical and physical placement"""

    def __init__(self, connectivity=None):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit):
        return dict(zip(list("q" + str(i) for i in range(circuit.nqubits)), range(circuit.nqubits)))


class Custom(Placer):
    """Define a custom initial qubit mapping.
    Attr:
        map (list or dict): List reporting or dict the circuit to chip qubit mapping,
        example [1,2,0] or {"q0":1, "q1":2, "q2":0} to assign the logical to physical qubit mapping.
    """

    def __init__(self, map, connectivity=None):
        self.connectivity = connectivity
        self.map = map

    def __call__(self, circuit=None):
        if isinstance(self.map, dict):
            if not check_mapping_consistency(self.map):
                raise_error(ValueError)
            return self.map
        elif isinstance(self.map, list):
            return dict(zip(list("q" + str(i) for i in range(len(self.map))), self.map))
        else:
            raise_error(TypeError, "Use dict or list to define mapping.")


class Subgraph(Placer):
    """
    Subgraph isomorphism qubit placer,
        NP-complete it can take a long time for large circuits.
        This initialization method may fail for very short circuits.
    """

    def __init__(self, connectivity):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit):
        # TODO fix networkx.GM.mapping for small subgraphs
        circuit_repr = create_circuit_repr(circuit)
        if len(circuit_repr) < 3:
            raise_error(ValueError, "Circuit must contain at least two two qubit gates to implement subgraph placement")
        H = nx.Graph()
        H.add_nodes_from([i for i in range(self.connectivity.number_of_nodes())])
        GM = nx.algorithms.isomorphism.GraphMatcher(self.connectivity, H)
        i = 0
        H.add_edge(circuit_repr[i][0], circuit_repr[i][1])
        while GM.subgraph_is_monomorphic() == True:
            result = GM
            i += 1
            H.add_edge(circuit_repr[i][0], circuit_repr[i][1])
            GM = nx.algorithms.isomorphism.GraphMatcher(self.connectivity, H)
            if self.connectivity.number_of_edges() == H.number_of_edges() or i == len(circuit_repr) - 1:
                keys = list(result.mapping.keys())
                keys.sort()
                return {i: result.mapping[i] for i in keys}
        keys = list(result.mapping.keys())
        keys.sort()
        return {i: result.mapping[i] for i in keys}


class Random(Placer):
    """
    Random initialization with greedy policy, let a maximum number of 2-qubit
        gates can be applied without introducing any SWAP gate
    """

    def __init__(self, connectivity, samples=100):
        self.connectivity = connectivity
        self.samples = samples

    def __call__(self, circuit):
        circuit_repr = create_circuit_repr(circuit)
        nodes = self.connectivity.number_of_nodes()
        keys = list(self.connectivity.nodes())
        final_mapping = dict(zip(keys, list(range(nodes))))
        final_graph = nx.relabel_nodes(self.connectivity, final_mapping)
        final_cost = self.cost(final_graph, circuit_repr)
        for _ in range(self.samples):
            mapping = dict(zip(keys, random.sample(range(nodes), nodes)))
            graph = nx.relabel_nodes(self.connectivity, mapping)
            cost = self.cost(graph, circuit_repr)
            if cost == 0:
                return mapping
            if cost < final_cost:
                final_graph = graph
                final_mapping = mapping
                final_cost = cost
        return final_mapping

    @staticmethod
    def cost(graph, circuit_repr):
        """
        Args:
            graph (networkx.Graph): current hardware qubit mapping.
            circuit_repr (list): circuit representation.

        Returns:
            (int): lengh of the reduced circuit.
        """
        new_circuit = circuit_repr.copy()
        while new_circuit != [] and (new_circuit[0][0], new_circuit[0][1]) in graph.edges():
            del new_circuit[0]
        return len(new_circuit)


# TODO
class Backpropagation(Placer):
    """
    Place qubits based on the algorithm proposed in
    https://doi.org/10.48550/arXiv.1809.02573
    """

    def __init__(self, connectivity, routing_algorithm, iterations=1, max_lookahead_gates=None):
        self.connectivity = connectivity
        self._routing = routing_algorithm
        self._iterations = iterations
        self._max_gates = max_lookahead_gates

    def __call__(self, circuit):
        # Start with trivial placement
        self._circuit_repr = create_circuit_repr(circuit)
        initial_placement = dict(zip(list("q" + str(i) for i in range(circuit.nqubits)), range(circuit.nqubits)))
        for _ in range(self._iterations):
            final_placement = self.forward_step(initial_placement)
            initial_placement = self.backward_step(final_placement)
        return initial_placement

    # TODO: requires block circuit
    def forward_step(self, initial_placement):
        return initial_placement

    # TODO: requires block circuit
    def backward_step(self, final_placement):
        # TODO: requires block circuit
        return final_placement
