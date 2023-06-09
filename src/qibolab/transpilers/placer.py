import random

import networkx as nx
from qibo.config import raise_error
from qibo.models import Circuit

from qibolab.transpilers.abstract import Placer, create_circuit_repr


class PlacementError(Exception):
    """Raise for an error in the qubit placement"""


def assert_placement(circuit: Circuit, layout: dict) -> bool:
    """Check if layout is correct and matches the number of qubits of the circuit.

    Args:
        circuit (qibo.models.Circuit): Circuit model to check.
        layout (dict): physical to logical qubit mapping.

    Raise PlacementError if the following conditions are not satisfied:
        - layout is written in the correct form.
        - layout matches the number of qubits in the circuit.
    """
    assert_mapping_consistency(layout)
    if circuit.nqubits > len(layout):
        raise PlacementError("Layout can't be used on circuit. The circuit requires more qubits.")
    if circuit.nqubits < len(layout):
        raise PlacementError("Layout can't be used on circuit. Ancillary extra qubits need to be added to the circuit.")


def assert_mapping_consistency(layout):
    """Check if layout is correct.

    Args:
        layout (dict): physical to logical qubit mapping.

    Raise PlacementError if layout is not written in the correct form.
    """
    values = sorted(layout.values())
    keys = list(layout.keys())
    ref_keys = ["q" + str(i) for i in range(len(keys))]
    if keys != ref_keys:
        raise PlacementError("Some physical qubits in the layout may be missing or duplicated.")
    if values != list(range(len(values))):
        raise PlacementError("Some logical qubits in the layout may be missing or duplicated.")


class Trivial(Placer):
    """Place qubits according to the following simple notation: {'q0' : 0, 'q1' : 1, ..., 'qn' : n}.

    Attributes:
        connectivity (networkx.Graph): chip connectivity.
    """

    def __init__(self, connectivity=None):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit):
        """Find the trivial placement for the circuit.

        Args:
            circuit (qibo.models.Circuit): Circuit model to check.
        """
        return dict(zip(list("q" + str(i) for i in range(circuit.nqubits)), range(circuit.nqubits)))


class Custom(Placer):
    """Define a custom initial qubit mapping.

    Attributes:
        map (list or dict): physical to logical qubit mapping,
        example [1,2,0] or {"q0":1, "q1":2, "q2":0} to assign the
        physical qubits 0;1;2 to the logical qubits 1;2;0 respectively.
        connectivity (networkx.Graph): chip connectivity.
    """

    def __init__(self, map, connectivity=None, verbose=False):
        self.connectivity = connectivity
        self.map = map

    def __call__(self, circuit=None):
        """Return the custom placement if it can be applied to the given circuit (if given).

        Args:
            circuit (qibo.models.Circuit): Circuit to be transpiled.
        """
        if isinstance(self.map, dict):
            pass
        elif isinstance(self.map, list):
            self.map = dict(zip(list("q" + str(i) for i in range(len(self.map))), self.map))
        else:
            raise_error(TypeError, "Use dict or list to define mapping.")
        if circuit is not None:
            assert_placement(circuit, self.map)
        else:
            assert_mapping_consistency(self.map)
        return self.map


class Subgraph(Placer):
    """
    Subgraph isomorphism qubit placer, NP-complete it can take a long time
    for large circuits. This initialization method may fail for very short circuits.

    Attributes:
        connectivity (networkx.Graph): chip connectivity.
    """

    def __init__(self, connectivity):
        self.connectivity = connectivity

    def __call__(self, circuit: Circuit):
        """Find the initial layout of the given circuit using subgraph isomorphism.

        Args:
            circuit (qibo.models.Circuit): Circuit to be transpiled.
        """
        circuit_repr = create_circuit_repr(circuit)
        if len(circuit_repr) < 3:
            raise_error(
                ValueError, "Circuit must contain at least two two qubit gates to implement subgraph placement."
            )
        circuit_subgraph = nx.Graph()
        circuit_subgraph.add_nodes_from([i for i in range(self.connectivity.number_of_nodes())])
        matcher = nx.algorithms.isomorphism.GraphMatcher(self.connectivity, circuit_subgraph)
        i = 0
        circuit_subgraph.add_edge(circuit_repr[i][0], circuit_repr[i][1])
        while matcher.subgraph_is_monomorphic() == True:
            result = matcher
            i += 1
            circuit_subgraph.add_edge(circuit_repr[i][0], circuit_repr[i][1])
            matcher = nx.algorithms.isomorphism.GraphMatcher(self.connectivity, circuit_subgraph)
            if self.connectivity.number_of_edges() == circuit_subgraph.number_of_edges() or i == len(circuit_repr) - 1:
                keys = list(result.mapping.keys())
                keys.sort()
                return {i: result.mapping[i] for i in keys}
        keys = list(result.mapping.keys())
        keys.sort()
        return {i: result.mapping[i] for i in keys}


class Random(Placer):
    """
    Random initialization with greedy policy, let a maximum number of 2-qubit
    gates can be applied without introducing any SWAP gate.

    Attributes:
        connectivity (networkx.Graph): chip connectivity.
        samples (int): number of initial random layouts tested.
    """

    def __init__(self, connectivity, samples=100):
        self.connectivity = connectivity
        self.samples = samples

    def __call__(self, circuit):
        """Find an initial layout of the given circuit using random greedy algorithm.

        Args:
            circuit (qibo.models.Circuit): Circuit to be transpiled.
        """
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
        Compute the cost associated to an initial layout as the lengh of the reduced circuit.

        Args:
            graph (networkx.Graph): current hardware qubit mapping.
            circuit_repr (list): circuit representation.

        Returns:
            (int): lengh of the reduced circuit.
        """
        total_len = len(circuit_repr)
        for allowed, gate in enumerate(circuit_repr):
            if (gate) not in graph.edges():
                break
        return total_len - allowed


# TODO: requires block decomposition
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
