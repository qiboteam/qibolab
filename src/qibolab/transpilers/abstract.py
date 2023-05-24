from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from qibo.backends import NumpyBackend
from qibo.config import log, raise_error
from qibo.models import Circuit


class Placer(ABC):
    """A placer implments the initial logical-physical qubit mapping"""

    @abstractmethod
    def place(self, circuit: Circuit) -> dict:
        """Find initial qubit mapping

        Args:
            circuit (:class:`qibo.models.Circuit`): circuit to be mapped.

        Returns:
            initial_layout (dict): dictionary containing the initial logical to physical qubit mapping
        """

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

    def create_circuit_repr(self, circuit):
        """Translate qibo circuit into a list of two qubit gates to be used by the transpiler.

        Args:
            circuit (:class:`qibo.models.Circuit`): circuit to be transpiled.

        Returns:
            translated_circuit (list): list containing qubits targeted by two qubit gates
        """
        translated_circuit = []
        for index, gate in enumerate(circuit.queue):
            if len(gate.qubits) == 2:
                gate_qubits = list(gate.qubits)
                gate_qubits.sort()
                gate_qubits.append(index)
                translated_circuit.append(gate_qubits)
            if len(gate.qubits) >= 3:
                raise_error(ValueError, "Gates targeting more than 2 qubits are not supported")
        return translated_circuit


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
            transpiled_circuit, _ = self.transpile(original_circuit)

        backend = NumpyBackend()
        target_state = backend.execute_circuit(original_circuit).state()
        final_state = backend.execute_circuit(transpiled_circuit).state()
        fidelity = np.abs(np.dot(np.conj(target_state), final_state))
        np.testing.assert_allclose(fidelity, 1.0)
        log.info("Transpiler test passed.")

    @abstractmethod
    def transpile(self, circuit: Circuit) -> Tuple[Circuit, List[int]]:
        """Apply the transpiler transformation on a given circuit.

        Args:
            circuit (:class:`qibo.models.Circuit`): Circuit to transpile.

        Retruns:
            circuit (:class:`qibo.models.Circuit`): Circuit after transpilation.
            qubit_map (list): Order of qubits in the transpiled circuit.
        """
        # TODO: Maybe use __call__? (simone: seems a good idea)

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
