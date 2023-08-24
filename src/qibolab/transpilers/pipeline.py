import networkx as nx
import numpy as np
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibolab.native import NativeType
from qibolab.transpilers.abstract import Optimizer, Placer, Router, Unroller
from qibolab.transpilers.optimizer import Preprocessing
from qibolab.transpilers.placer import Trivial, assert_placement
from qibolab.transpilers.router import (
    ConnectivityError,
    ShortestPaths,
    assert_connectivity,
)
from qibolab.transpilers.unroller import (
    DecompositionError,
    NativeGates,
    assert_decomposition,
)


class TranspilerPipelineError(Exception):
    """Raise when an error occurs in the transpiler pipeline"""


# TODO: rearrange qubits based on the final qubit map (or it will not work for routed circuit)
def assert_cirucuit_equivalence(original_circuit: Circuit, transpiled_circuit: Circuit):
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


def assert_transpiling(
    circuit: Circuit,
    connectivity: nx.Graph,
    initial_layout: dict,
    final_layout: dict,
    native_gates: NativeType = NativeType.CZ,
):
    """Check that all transpiler passes have been executed correctly"""
    assert_connectivity(circuit=circuit, connectivity=connectivity)
    assert_decomposition(circuit=circuit, two_qubit_natives=native_gates)
    assert_placement(circuit=circuit, layout=initial_layout)
    assert_placement(circuit=circuit, layout=final_layout)


class Passes:
    """Define a transpiler pipeline consisting of smaller transpiler steps that are applied sequentially:

    Args:
        passes (list): list of passes to be applied sequentially,
            if None default transpiler will be used, it requires hardware connectivity.
        connectivity (nx.Graph): hardware qubit connectivity.
    """

    def __init__(self, passes: list = None, connectivity: nx.Graph = None, native_gates: NativeType = NativeType.CZ):
        self.native_gates = native_gates
        if passes is None:
            self.passes = self.default(connectivity)
        else:
            self.passes = passes
        self.connectivity = connectivity

    def default(self, connectivity: nx.Graph):
        """Return the default transpiler pipeline for the required hardware connectivity."""
        if not isinstance(connectivity, nx.Graph):
            raise TranspilerPipelineError("Define the hardware chip connectivity to use default transpiler")
        default_passes = []
        # preprocessing
        default_passes.append(Preprocessing(connectivity=connectivity))
        # default placer pass
        default_passes.append(Trivial(connectivity=connectivity))
        # default router pass
        default_passes.append(ShortestPaths(connectivity=connectivity))
        # default unroller pass
        default_passes.append(NativeGates(two_qubit_natives=self.native_gates))
        return default_passes

    def __call__(self, circuit):
        self.initial_layout = None
        final_layout = None
        for transpiler_pass in self.passes:
            if isinstance(transpiler_pass, Optimizer):
                circuit = transpiler_pass(circuit)
            elif isinstance(transpiler_pass, Placer):
                if self.initial_layout == None:
                    self.initial_layout = transpiler_pass(circuit)
                else:
                    raise TranspilerPipelineError("You are defining more than one placer pass.")
            elif isinstance(transpiler_pass, Router):
                if self.initial_layout is not None:
                    circuit, final_layout = transpiler_pass(circuit, self.initial_layout)
                else:
                    raise TranspilerPipelineError("Use a placement pass before routing.")
            elif isinstance(transpiler_pass, Unroller):
                circuit = transpiler_pass(circuit)
            else:
                raise TranspilerPipelineError("Unrecognised transpiler pass: ", transpiler_pass)
        return circuit, final_layout

    def is_satisfied(self, circuit):
        """Return True if the circuit respects the hardware connectivity and native gates, False otherwise.

        Args:
            circuit (qibo.models.Circuit): circuit to be checked.
            native_gates (NativeType): two qubit native gates.
        """
        try:
            assert_connectivity(circuit=circuit, connectivity=self.connectivity)
            assert_decomposition(circuit=circuit, two_qubit_natives=self.native_gates)
            return True
        except ConnectivityError:
            return False
        except DecompositionError:
            return False

    def get_initial_layout(self):
        """Return initial qubit layout"""
        return self.initial_layout
