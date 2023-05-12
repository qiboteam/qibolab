import random
from enum import Enum, auto

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from more_itertools import pairwise
from qibo import gates
from qibo.config import log, raise_error
from qibo.models import Circuit

from qibolab.transpilers.gate_decompositions import TwoQubitNatives, translate_gate

DEFAULT_INIT_SAMPLES = 100


class QubitInitMethod(Enum):
    """A class to define the initial qubit mapping methods."""

    greedy = auto()
    subgraph = auto()
    custom = auto()


class Transpiler:
    """A class to perform initial qubit mapping and connectivity matching.

    Properties:
        connectivity (networkx.graph): chip connectivity.
        init_method (str or QubitInitMethod): initial qubit mapping method.
        init_samples (int): number of random qubit initializations for greedy initial qubit mapping.
        two_qubit_natives (TwoQubitNatives or str): two qubit gate/s that can be implemented by the hardware.
        sampling_split (float): fraction of paths tested (between 0 and 1).

    Attributes:
        _circuit_repr (list): quantum circuit represented as a list (only 2 qubit gates).
        _mapping (dict): circuit to physical qubit mapping during transpiling.
        _graph (networkx.graph): qubit mapped as nodes of the connectivity graph.
        _qubit_map (np.array): circuit to physical qubit mapping during transpiling as vector.
        _circuit_position (int): position in the circuit.
        _added_swaps (int): number of swaps added to the circuit to match connectivity.
    """

    def __init__(
        self, connectivity, init_method="greedy", init_samples=None, two_qubit_natives="CZ", sampling_split=1.0
    ):
        self.connectivity = connectivity
        self.init_method = init_method
        if self.init_method is QubitInitMethod.greedy and init_samples is None:
            init_samples = DEFAULT_INIT_SAMPLES
        self.init_samples = init_samples
        self.two_qubit_natives = two_qubit_natives
        self.sampling_split = sampling_split

        self._circuit_repr = None
        self._mapping = None
        self._graph = None
        self._qubit_map = None
        self._transpiled_circuit = None
        self._circuit_position = 0
        self._added_swaps = 0

    def transpile(self, circuit, fuse_one_qubit=False, fusion_algorithm=False):
        """Full transpilation, match connectivity and translation into native gates.

        Args:
            circuit (:class:`qibo.models.Circuit`): circuit to be transpiled.
            fuse_one_qubit (bool): Fuse two or more one qubit gates in sequence
            fusion_algorithm (bool): Try to reduce the number of SWAP in the transpiler by using qibo fusion algorithm

        Returns:
            transpiled_circuit (qibo.Circuit): circut mapped to hardware topology with only native gates.
            final_mapping (dict): logical to physical qubit mapping after the execution of the circuit.
                key (int) is the logical qubit, value (int) is the physical qubit.
            init_mapping (dict): logical to physical qubit mapping before the execution of the circuit
                key (int) is the logical qubit, value (int) is the physical qubit.
            added_swaps (int): number of swap gates added.
        """

        # TODO: ask Stavros what this part of the code do and if it is useful in this case
        if fusion_algorithm:
            # Re-arrange gates using qibo's fusion algorithm
            # this may reduce number of SWAPs when fixing for connectivity
            fcircuit = circuit.fuse(max_qubits=2)
            new = type(circuit)(circuit.nqubits)
            for fgate in fcircuit.queue:
                if isinstance(fgate, gates.FusedGate):
                    new.add(fgate.gates)
                else:
                    new.add(fgate)
            circuit = new

        # Match connectivity
        mapped_circuit, final_mapping, init_mapping, added_swaps = self.match_connectivity(circuit)
        # Two-qubit gates to native
        new = translate_circuit(mapped_circuit, two_qubit_natives=self.two_qubit_natives, translate_single_qubit=False)
        # Optional: fuse one-qubit gates to reduce circuit depth
        if fuse_one_qubit:
            new = new.fuse(max_qubits=1)
        # One-qubit gates to native
        transpiled_circuit = translate_circuit(
            new, two_qubit_natives=self.two_qubit_natives, translate_single_qubit=True
        )
        return transpiled_circuit, final_mapping, init_mapping, added_swaps

    def match_connectivity(self, circuit):
        """Qubit mapping initialization and circuit connectivity matching.

        Args:
            circuit (:class:`qibo.models.Circuit`): circuit to be matched to hardware connectivity.

        Returns:
            hardware_mapped_circuit (qibo.Circuit): circut mapped to hardware topology.
            final_mapping (dict): final qubit mapping.
            init_mapping (dict): initial qubit mapping.
            added_swaps (int): number of swap gates added.
        """
        self._circuit_position = 0
        self._added_swaps = 0
        self.create_circuit_repr(circuit)
        keys = list(self._connectivity.nodes())
        if self._init_method is QubitInitMethod.greedy:
            self.greedy_init()
        elif self._init_method is QubitInitMethod.subgraph:
            if len(self._circuit_repr) < 2:
                raise_error(
                    ValueError,
                    "The circuit must contain at least two two-qubit gates in order to apply subgraph initialization",
                )
            self.subgraph_init()
        elif self._init_method is QubitInitMethod.custom:
            self._mapping = dict(zip(keys, self._mapping.values()))
            self._graph = nx.relabel_nodes(self._connectivity, self._mapping)
        # Inverse permutation
        init_qubit_map = np.argsort(list(self._mapping.values()))
        init_mapping = dict(zip(keys, init_qubit_map))
        self._qubit_map = np.sort(init_qubit_map)
        self.init_circuit(circuit)
        self.first_transpiler_step(circuit)
        while len(self._circuit_repr) != 0:
            self.transpiler_step(circuit)
        final_mapping = {key: init_qubit_map[self._qubit_map[i]] for i, key in enumerate(keys)}
        hardware_mapped_circuit = self.init_mapping_circuit(self._transpiled_circuit, init_qubit_map)
        return hardware_mapped_circuit, final_mapping, init_mapping, self._added_swaps

    def transpiler_step(self, qibo_circuit):
        """Transpilation step. Find new mapping, add swap gates and apply gates that can be run with this configuration.

        Args:
            qibo_circuit (:class:`qibo.models.Circuit`): circuit to be transpiled.
        """
        len_2q_circuit = len(self._circuit_repr)
        path, meeting_point = self.relocate()
        self.add_swaps(path, meeting_point)
        self.update_qubit_map()
        self.add_gates(qibo_circuit, len_2q_circuit - len(self._circuit_repr))

    def first_transpiler_step(self, qibo_circuit):
        """First transpilation step. Apply gates that can be run with the initial qubit mapping.

        Args:
            qibo_circuit (:class:`qibo.models.Circuit`): circuit to be transpiled.
        """
        len_2q_circuit = len(self._circuit_repr)
        self._circuit_repr = self.reduce(self._graph)
        self.add_gates(qibo_circuit, len_2q_circuit - len(self._circuit_repr))

    def custom_qubit_mapping(self, map):
        """Define a custom initial qubit mapping.

        Args:
            map (list): List reporting the circuit to chip qubit mapping,
            example [1,2,0] to assign the logical to physical qubit mapping.
        """
        self.init_method = QubitInitMethod.custom
        self._mapping = dict(zip(range(len(map)), map))

    @property
    def connectivity(self):
        return self._connectivity

    @connectivity.setter
    def connectivity(self, connectivity):
        """Set the hardware chip connectivity.

        Args:
            connectivity (networkx graph): define connectivity.
        """

        if isinstance(connectivity, nx.Graph):
            self._connectivity = connectivity
        else:
            raise_error(TypeError, "Use networkx graph for custom connectivity")

    @property
    def sampling_split(self):
        return self._sampling_split

    @sampling_split.setter
    def sampling_split(self, sampling_split):
        """Set the sampling split.

        Args:
            sampling_split (float): define fraction of shortest path tested.
        """

        if sampling_split > 0.0 and 1.0 >= sampling_split:
            self._sampling_split = sampling_split
        else:
            raise_error(ValueError, "Sampling_split must be set greater than 0 and less or equal 1")

    def draw_connectivity(self):  # pragma: no cover
        """Show connectivity graph."""
        pos = nx.spectral_layout(self._connectivity)
        nx.draw(self._connectivity, pos=pos, with_labels=True)
        plt.show()

    @property
    def init_method(self):
        return self._init_method

    @init_method.setter
    def init_method(self, init_method):
        """Set the initial mapping method for the transpiler.

        Args:
            init_method (str): Initial mapping method ("greedy" or "subgraph").
        """
        if isinstance(init_method, str):
            init_method = QubitInitMethod[init_method]
        self._init_method = init_method

    @property
    def two_qubit_natives(self):
        return self._two_qubit_natives

    @two_qubit_natives.setter
    def two_qubit_natives(self, two_qubit_natives):
        """Set the native hardware two qubit gates.

        Args:
            two_qubit_natives (TwoQubitNatives or str):
        """
        if isinstance(two_qubit_natives, str):
            two_qubit_natives = TwoQubitNatives[two_qubit_natives]
        self._two_qubit_natives = two_qubit_natives

    @property
    def init_samples(self):
        return self._init_samples

    @init_samples.setter
    def init_samples(self, init_samples):
        """Set the initial mapping method for the transpiler.

        Args:
            init_samples (int): Number of random qubit mapping samples to be tested
            (required only for "greedy" initialization).
        """
        if init_samples is not None:
            self.init_method = QubitInitMethod.greedy
        self._init_samples = init_samples

    def create_circuit_repr(self, qibo_circuit):
        """Translate qibo circuit into a list of two qubit gates to be used by the transpiler.

        Args:
            qibo_circuit (:class:`qibo.models.Circuit`): circuit to be transpiled.
        """
        translated_circuit = []
        for index, gate in enumerate(qibo_circuit.queue):
            if len(gate.qubits) == 2:
                gate_qubits = list(gate.qubits)
                gate_qubits.sort()
                gate_qubits.append(index)
                translated_circuit.append(gate_qubits)
            if len(gate.qubits) >= 3:
                raise_error(ValueError, "ERROR do not use gates acting on more than 2 qubits")
        self._circuit_repr = translated_circuit

    def reduce(self, graph):
        """Reduce the circuit, delete a 2-qubit gate if it can be applied on the current configuration.

        Args:
            graph (networkx.Graph): current hardware qubit mapping.

        Returns:
            new_circuit (list): reduced circuit.
        """
        new_circuit = self._circuit_repr.copy()
        while new_circuit != [] and (new_circuit[0][0], new_circuit[0][1]) in graph.edges():
            del new_circuit[0]
        return new_circuit

    def subgraph_init(self):
        # TODO fix networkx.GM.mapping for small subgraphs
        """Subgraph isomorphism initialization, NP-complete it can take a long time for large circuits.
        This initialization method may fail for very short circuits.
        """
        H = nx.Graph()
        H.add_nodes_from([i for i in range(self._connectivity.number_of_nodes())])
        GM = nx.algorithms.isomorphism.GraphMatcher(self._connectivity, H)
        i = 0
        H.add_edge(self._circuit_repr[i][0], self._circuit_repr[i][1])
        while GM.subgraph_is_monomorphic() == True:
            result = GM
            i += 1
            H.add_edge(self._circuit_repr[i][0], self._circuit_repr[i][1])
            GM = nx.algorithms.isomorphism.GraphMatcher(self._connectivity, H)
            if self._connectivity.number_of_edges() == H.number_of_edges() or i == len(self._circuit_repr) - 1:
                G = nx.relabel_nodes(self._connectivity, result.mapping)
                keys = list(result.mapping.keys())
                keys.sort()
                self._graph = G
                self._mapping = {i: result.mapping[i] for i in keys}
                return
        G = nx.relabel_nodes(self._connectivity, result.mapping)
        keys = list(result.mapping.keys())
        keys.sort()
        self._graph = G
        self._mapping = {i: result.mapping[i] for i in keys}

    def greedy_init(self):
        """Initialize the circuit with greedy algorithm let a maximum number of 2-qubit
        gates can be applied without introducing any SWAP gate"""
        nodes = self._connectivity.number_of_nodes()
        keys = list(self._connectivity.nodes())
        final_mapping = dict(zip(keys, list(range(nodes))))
        final_graph = nx.relabel_nodes(self._connectivity, final_mapping)
        final_cost = len(self.reduce(final_graph))
        for _ in range(self._init_samples):
            mapping = dict(zip(keys, random.sample(range(nodes), nodes)))
            graph = nx.relabel_nodes(self._connectivity, mapping)
            cost = len(self.reduce(graph))
            if cost == 0:
                self._graph = graph
                self._mapping = mapping
                return
            if cost < final_cost:
                final_graph = graph
                final_mapping = mapping
                final_cost = cost
        self._graph = final_graph
        self._mapping = final_mapping

    def map_list(self, path):
        """Return all possible walks of qubits, or a fraction, for a given path.

        Args:
            path (list): path to move qubits.

        Returns:
            mapping_list (list): all possible walks of qubits, or a fraction of them based on self.sampling_split, for a given path.
            meeting_point_list (list): qubit meeting point for each path.
        """
        path_ends = [path[0], path[-1]]
        path_middle = path[1:-1]
        mapping_list = []
        meeting_point_list = []
        test_paths = list(range(len(path) - 1))
        if self.sampling_split != 1.0:
            test_paths = np.random.choice(
                test_paths, size=int(np.ceil(len(test_paths) * self.sampling_split)), replace=False
            )
        for i in test_paths:
            values = path_middle[:i] + path_ends + path_middle[i:]
            mapping = dict(zip(path, values))
            mapping_list.append(mapping)
            meeting_point_list.append(i)
        return mapping_list, meeting_point_list

    def relocate(self):
        """A small greedy algorithm to decide which path to take, and how qubits should walk.

        Returns:
            final_path (list): best path to move qubits.
            meeting_point (int): qubit meeting point in the path.
        """
        nodes = self._graph.number_of_nodes()
        circuit = self.reduce(self._graph)
        final_circuit = circuit
        keys = list(range(nodes))
        final_graph = self._graph
        final_mapping = dict(zip(keys, keys))
        # Consider all shortest paths
        path_list = [p for p in nx.all_shortest_paths(self._graph, source=circuit[0][0], target=circuit[0][1])]
        self._added_swaps += len(path_list[0]) - 2
        # Here test all paths
        for path in path_list:
            # map_list uses self.sampling_split
            list_, meeting_point_list = self.map_list(path)
            for j, mapping in enumerate(list_):
                new_graph = nx.relabel_nodes(self._graph, mapping)
                new_circuit = self.reduce(new_graph)
                # Greedy looking for the optimal path and the optimal walk on this path
                if len(new_circuit) < len(final_circuit):
                    final_graph = new_graph
                    final_circuit = new_circuit
                    final_mapping = mapping
                    final_path = path
                    meeting_point = meeting_point_list[j]
        self._graph = final_graph
        self._mapping = final_mapping
        self._circuit_repr = final_circuit
        return final_path, meeting_point

    def init_circuit(self, qibo_circuit):
        """Initialize the transpiled circuit

        Args:
            Args: qibo_circuit (:class:`qibo.models.Circuit`): circuit to be transpiled.
        """
        nodes = self._connectivity.number_of_nodes()
        qubits = qibo_circuit.nqubits
        if qubits > nodes:
            raise_error(ValueError, "There are not enough physical qubits in the hardware to map the circuit")
        elif qubits == nodes:
            new_circuit = Circuit(nodes)
        else:
            log.warning(
                "You are using more physical qubits than required by the circuit, some qubits will be added to the circuit"
            )
            new_circuit = Circuit(nodes)
        self._transpiled_circuit = new_circuit

    def init_mapping_circuit(self, circuit, qubit_map):
        """Initial qubit mapping of the transpiled qibo circuit

        Args:
            circuit (:class:`qibo.models.Circuit`): transpiled qibo circuit.
            qubit_map (np.array): initial qubit mapping.

        Returns:
            new_circuit (:class:`qibo.models.Circuit`): transpiled circuit mapped with initial qubit mapping.
        """
        new_circuit = Circuit(self._connectivity.number_of_nodes())
        for gate in circuit.queue:
            new_circuit.add(gate.on_qubits({q: qubit_map[q] for q in gate.qubits}))
        return new_circuit

    def add_gates(self, qibo_circuit, matched_gates):
        """Add one and two qubit gates to transpiled circuit until connectivity is matched

        Args:
            qibo_circuit (:class:`qibo.models.Circuit`): circuit to be transpiled.
            matched_gates (int): number of two qubit gates that can be applied with the current qubit mapping
        """
        index = 0
        while self._circuit_position < len(qibo_circuit.queue):
            gate = qibo_circuit.queue[self._circuit_position]
            if len(gate.qubits) == 1:
                self._transpiled_circuit.add(gate.on_qubits({gate.qubits[0]: self._qubit_map[gate.qubits[0]]}))
                self._circuit_position += 1
            else:
                index += 1
                if index == matched_gates + 1:
                    break
                else:
                    self._transpiled_circuit.add(
                        gate.on_qubits(
                            {
                                gate.qubits[0]: self._qubit_map[gate.qubits[0]],
                                gate.qubits[1]: self._qubit_map[gate.qubits[1]],
                            }
                        )
                    )
                    self._circuit_position += 1

    def add_swaps(self, path, meeting_point):
        """Add swaps to the transpiled circuit to move qubits

        Args:
            path (list): path to move qubits.
            meeting_point (int): qubit meeting point in the path.
        """
        forward = path[0 : meeting_point + 1]
        backward = list(reversed(path[meeting_point + 1 :]))
        if len(forward) > 1:
            for f1, f2 in pairwise(forward):
                self._transpiled_circuit.add(gates.SWAP(self._qubit_map[f1], self._qubit_map[f2]))
        if len(backward) > 1:
            for b1, b2 in pairwise(backward):
                self._transpiled_circuit.add(gates.SWAP(self._qubit_map[b1], self._qubit_map[b2]))

    def update_qubit_map(self):
        """Update the qubit mapping after adding swaps"""
        old_mapping = self._qubit_map.copy()
        for key, value in self._mapping.items():
            self._qubit_map[value] = old_mapping[key]


def translate_circuit(circuit, two_qubit_natives, translate_single_qubit=False):
    """Translates a circuit to native gates.

    Args:
        circuit (qibo.models.Circuit): Circuit model to translate into native gates.
        two_qubit_natives (list): List of two qubit native gates
            supported by the quantum hardware ("CZ" and/or "iSWAP").
        translate_single_qubit (bool):

    Returns:
        new (qibo.models.Circuit): Equivalent circuit with native gates.
    """
    new = type(circuit)(circuit.nqubits)
    for gate in circuit.queue:
        if len(gate.qubits) > 1 or translate_single_qubit:
            new.add(translate_gate(gate, two_qubit_natives))
        else:
            new.add(gate)
    return new


def can_execute(circuit: Circuit, two_qubit_natives: TwoQubitNatives, connectivity: nx.Graph, verbose=True):
    """Checks if a circuit can be executed on Hardware.

    Args:
        circuit (qibo.models.Circuit): Circuit model to check.
        two_qubit_natives (TwoQubitNatives): two qubit gate/s that can be implemented by the hardware.
        connectivity (networkx.graph): chip connectivity.
        verbose (bool): If ``True`` it prints debugging log messages.

    Returns ``True`` if the following conditions are satisfied:
        - Circuit does not contain more than two-qubit gates.
        - All one-qubit gates are I, Z, RZ or U3.
        - All two-qubit gates are CZ or iSWAP based on two_qubit_natives.
        - Circuit matches connectivity.
    otherwise returns ``False``.
    """

    # pring messages only if ``verbose == True``
    vlog = lambda msg: log.info(msg) if verbose else lambda msg: None
    for gate in circuit.queue:
        if isinstance(gate, gates.M):
            continue

        if len(gate.qubits) == 1:
            if not isinstance(gate, (gates.I, gates.Z, gates.RZ, gates.U3)):
                vlog(f"{gate.name} is not a single qubit native gate.")
                return False

        elif len(gate.qubits) == 2:
            try:
                if not (TwoQubitNatives.from_gate(gate) in two_qubit_natives):
                    vlog(f"{gate.name} is not in two_qubit_native.")
                    return False
            except ValueError:
                vlog(f"{gate.name} cannot be used as a two qubit native gate.")
                return False
            if gate.qubits not in connectivity.edges:
                vlog("Circuit does not respect connectivity. " f"{gate.name} acts on {gate.qubits}.")
                return False
        else:
            vlog(f"{gate.name} acts on more than two qubits.")
            return False
    vlog("Circuit can be executed.")
    return True
