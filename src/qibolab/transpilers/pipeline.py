import networkx as nx
import numpy as np
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibolab.transpilers.abstract import Placer, Router
from qibolab.transpilers.fusion import Fusion, Rearrange
from qibolab.transpilers.placer import Trivial
from qibolab.transpilers.router import ShortestPaths
from qibolab.transpilers.unroller import NativeGates


# TODO: rearrange qubits based on the final qubit map (or it will not work for routed circuit)
def assert_transpilation_equivalence(original_circuit: Circuit, transpiled_circuit: Circuit):
    """Checks that the transpiled circuit agrees with the original using simulation.

    Args:
        original_circuit (qibo.models.Circuit): Original circuit.
        transpiled_circuit (qibo.models.Circuit): Transpiled circuit.
    """
    backend = NumpyBackend()
    target_state = backend.execute_circuit(original_circuit).state()
    final_state = backend.execute_circuit(transpiled_circuit).state()
    fidelity = np.abs(np.dot(np.conj(target_state), final_state))
    np.testing.assert_allclose(fidelity, 1.0)


class Complete:
    """Complete transpiler pipeline consisting of smaller transpiler steps that are applied sequentially:
    1) preprocessing
    2) initial qubit placement
    3) routing
    4) optimization
    5) gate translation

    Args:
        placer (transpilers.abstract.Placer): algorithm to define initial qubit mapping.
        router (transpilers.abstract.Router): algorithm to perform routing.
        preprocessing (transpilers.abstract.Optimizer): circuit optimization before connectivity matching.
        optimizer (transpilers.abstract.Optimizer): circuit optimization after connectivity matching.
        unroller (transpilers.abstract.Unroller): circuit translation to native gates.

        NB: preprocessing and optimizers can also be a list in order to perform several of these steps,
        in the future when hopefully we will have more than one optimization procedure.
    """

    def __init__(
        self,
        connectivity: nx.Graph,
        placer: Placer = Trivial,
        router: Router = ShortestPaths,
        preprocessing=Rearrange,
        optimizer=Fusion,
        unroller=NativeGates,
    ):
        self.placer = placer
        self.router = router
        self.preprocessing = preprocessing
        self.optimizer = optimizer
        self.unroller = unroller
        self.connectivity = connectivity

    def __call__(self, circuit):
        if self.preprocessing is not None:
            circuit = self.preprocessing(circuit)
        initial_layout = self.placer(circuit)
        routed_circuit, final_map = self.router(circuit, initial_layout)
        if self.optimizer is not None:
            routed_circuit = self.optimizer(routed_circuit)
        final_circuit = self.unroller(routed_circuit)
        return final_circuit, final_map


class ConnectivityMatch:
    """Only connectivty matching"""

    def __init__(self, connectivity: nx.Graph, placer: Placer = Trivial, router: Router = ShortestPaths):
        self.placer = placer
        self.router = router
        self.connectivity = connectivity

    def __call__(self, circuit):
        initial_layout = self.placer(circuit)
        routed_circuit, final_map = self.router(circuit, initial_layout)
        return routed_circuit, final_map


class Optimization:
    """Only optimization steps"""

    def __init__(self, optimizers=[Rearrange, Fusion]):
        self.optimizers = optimizers

    def __call__(self, circuit):
        for optimizer in self.optimizers:
            circuit = optimizer(circuit)
        return circuit
