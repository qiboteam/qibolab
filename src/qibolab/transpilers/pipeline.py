import networkx as nx
import numpy as np
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibolab.native import NativeType
from qibolab.transpilers.abstract import Optimizer, Placer, Router, Unroller
from qibolab.transpilers.placer import Trivial
from qibolab.transpilers.router import ShortestPaths
from qibolab.transpilers.unroller import NativeGates


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


class Passes:
    """Define a transpiler pipeline consisting of smaller transpiler steps that are applied sequentially:

    Args:
        passes (list): list of passes to be applied sequentially,
            if None default transpiler will be used, it requires hardware connectivity.
        connectivity (nx.Graph): hardware qubit connectivity.
    """

    def __init__(self, passes: list = None, connectivity: nx.Graph = None):
        if passes is None:
            self.passes = self.default(connectivity)
        else:
            self.passes = passes

    def default(self, connectivity: nx.Graph):
        """Return the default transpiler pipeline for the required hardware connectivity."""
        if not isinstance(connectivity, nx.Graph):
            raise TranspilerPipelineError("Define the hardware chip connectivity to use default transpiler")
        default_passes = []
        # default placer pass
        default_passes.append(Trivial(connectivity=connectivity))
        # default router pass
        default_passes.append(ShortestPaths(connectivity=connectivity))
        # default unroller pass
        default_passes.append(NativeGates(two_qubit_natives=NativeType.CZ))
        return default_passes

    def __call__(self, circuit):
        layout = None
        final_layout = None
        for transpiler_pass in self.passes:
            if isinstance(transpiler_pass, Optimizer):
                circuit = transpiler_pass(circuit)
            elif isinstance(transpiler_pass, Placer):
                if layout == None:
                    layout = transpiler_pass(circuit)
                else:
                    raise TranspilerPipelineError("You are defining more than one placer pass.")
            elif isinstance(transpiler_pass, Router):
                if self.layout is not None:
                    circuit, final_layout = transpiler_pass(circuit, layout)
                else:
                    raise TranspilerPipelineError("Use a placement pass before routing.")
            elif isinstance(transpiler_pass, Unroller):
                circuit = transpiler_pass(circuit)
            else:
                TranspilerPipelineError("Unrecognised transpiler pass: ", transpiler_pass)
        return circuit, final_layout
