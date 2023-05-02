from abc import ABC, abstractmethod

from qibo.models import Circuit


class AbstractTranspiler(ABC):
    @abstractmethod
    def is_satisfied(self, circuit: Circuit) -> bool:
        """Checks if the circuit already satisfies this transpiler's requirements."""

    @abstractmethod
    def transpile(self, circuit: Circuit) -> Circuit:
        """Apply the transpiler transformation on a given circuit.

        Args:
            circuit (qibo.models.Circuit): Circuit to transpile.

        Retruns:
            circuit (qibo.models.Circuit): Circuit after transpilation.
            qubit_map (list): Order of qubits in the transpiled circuit.
        """
        # TODO: Maybe use __call__?
