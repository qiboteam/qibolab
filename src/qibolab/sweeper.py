from dataclasses import dataclass
from typing import Optional

import numpy.typing as npt

from qibolab.pulses import PulseType


@dataclass
class Sweeper:
    parameter: str
    values: npt.NDArray
    pulses: Optional[list] = None
    qubits: Optional[list] = None

    @property
    def pulse_type(self) -> Optional[PulseType]:
        types = {p.type for p in self.pulses}
        if len(types) > 1:
            raise RuntimeError("Not homogeneous pulses")
        return next(iter(types), None)
