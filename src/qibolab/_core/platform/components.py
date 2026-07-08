from pydantic import Field

from ..instruments.abstract import InstrumentMap
from ..qubits import QubitMap
from ..serialize import Model

__all__ = ["Hardware"]


class Hardware(Model):
    """Part of the platform that specifies the hardware configuration."""

    instruments: InstrumentMap
    qubits: QubitMap
    couplers: QubitMap = Field(default_factory=dict)
