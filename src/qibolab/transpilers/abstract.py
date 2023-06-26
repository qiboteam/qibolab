from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from qibo.backends import NumpyBackend
from qibo.config import log, raise_error
from qibo.models import Circuit


def find_gates_qubits_pairs(circuit: Circuit):
    """Translate qibo circuit into a list of pairs of qubits to be used by the transpiler.

    Args:
        circuit (qibo.models.Circuit): circuit to be transpiled.

    Returns:
        translated_circuit (list): list containing qubits targeted by two qubit gates
    """
    translated_circuit = []
    for gate in circuit.queue:
        if len(gate.qubits) == 2:
            translated_circuit.append(sorted(gate.qubits))
        if len(gate.qubits) >= 3:
            raise_error(ValueError, "Gates targeting more than 2 qubits are not supported")
    return translated_circuit


class Placer(ABC):
    """A placer implements the initial logical-physical qubit mapping"""

    @abstractmethod
    def __init__(self, connectivity: nx.Graph, *args):
        self.connectivity = connectivity

    @abstractmethod
    def __call__(self, circuit: Circuit) -> dict:
        """Find initial qubit mapping

        Args:
            circuit (qibo.models.Circuit): circuit to be mapped.

        Returns:
            initial_layout (dict): dictionary containing the initial logical to physical qubit mapping.
        """


class Router(ABC):
    """A router implements the mapping of a circuit on a specific hardware"""

    @abstractmethod
    def __init__(self, connectivity: nx.Graph, *args):
        self.connectivity = connectivity

    @abstractmethod
    def __call__(self, circuit: Circuit, initial_layout: dict) -> Tuple[Circuit, dict]:
        """Match circuit to hardware connectivity

        Args:
            circuit (qibo.models.Circuit): circuit to be routed.
            initial_layout (dict): dictionary containing the initial logical to physical qubit mapping.

        Returns:
            matched_circuit (qibo.models.Circuit): routed circuit
            final_layout (dict): dictionary containing the final logical to physical qubit mapping.
        """


class Transpiler(ABC):
    """A transpiler is a transformation from a circuit to another circuit."""

    @abstractmethod
    def is_satisfied(self, circuit: Circuit) -> bool:
        """Checks if the circuit satisfies this transpiler's requirements.

        In some cases this is computationally easier to check than applying
        the transpilation transformations.
        """

    def check_execution(self, original_circuit: Circuit, transpiled_circuit: Optional[Circuit] = None):
        """Checks that the transpiled circuit agrees with the original using simulation.

        Args:
            original_circuit (:class:`qibo.models.Circuit`): Original circuit.
            transpiled_circuit (:class:`qibo.models.Circuit`): Transpiled circuit.
                If not given, it will be calculated using
                :meth:`qibolab.transpilers.abstract.Transpiler.transpile`.
        """
        if transpiled_circuit is None:
            transpiled_circuit, _ = self(original_circuit)

        backend = NumpyBackend()
        target_state = backend.execute_circuit(original_circuit).state()
        final_state = backend.execute_circuit(transpiled_circuit).state()
        fidelity = np.abs(np.dot(np.conj(target_state), final_state))
        np.testing.assert_allclose(fidelity, 1.0)
        log.info("Transpiler test passed.")

    @abstractmethod
    def __call__(self, circuit: Circuit) -> Tuple[Circuit, List[int]]:
        """Apply the transpiler transformation on a given circuit.

        Args:
            circuit (:class:`qibo.models.Circuit`): Circuit to transpile.

        Retruns:
            circuit (:class:`qibo.models.Circuit`): Circuit after transpilation.
            qubit_map (list): Order of qubits in the transpiled circuit.
        """
