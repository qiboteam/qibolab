from dataclasses import dataclass
from typing import Optional

import numpy.typing as npt


@dataclass
class Sweeper:

    parameter: str
    values: npt.NDArray
    pulses: Optional[list] = None
    qubits: Optional[list] = None
    # TODO: Change that to the platform default wait time
    wait_time: int = 0

    @property
    def pulse_type(self):
        if self.pulses is not None:
            pulse_type = self.pulses[0].type.name.lower()
            for pulse in self.pulses:
                assert pulse.type.name.lower() == pulse_type
            return pulse_type
