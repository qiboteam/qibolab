from abc import ABC, abstractmethod

from qibo.models import Circuit


class AbstractTranspiler(ABC):
    @abstractmethod
    def is_satisfied(self, circuit: Circuit) -> bool:
        """Checks if the circuit already satisfies this transpiler's requirements."""

    @abstractmethod
    def transpile(self, circuit: Circuit) -> Circuit:
        """Apply the transpiler transformation on a given circuit."""
        # TODO: Maybe use __call__?
