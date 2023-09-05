from dataclasses import dataclass, field
from typing import List, Optional, Union

from qibolab.channels import Channel
from qibolab.qubits import Qubit

CouplerId = Union[str, int]
"""Type for Coupler names."""


@dataclass
class Coupler:
    """Representation of a physical coupler.

    Coupler objects are instantiated by :class:`qibolab.platforms.platform.Platform`
    but they are passed to instrument designs in order to play pulses.

    Args:
        name (int, str): Qubit number or name.

        flux (:class:`qibolab.platforms.utils.Channel`): Channel used to
            send flux pulses to the qubit.
        Other characterization parameters for the coupler, loaded from the runcard.
    """

    name: CouplerId

    # TODO: I think is not needed
    frequency: int = 0

    sweetspot: float = 0

    flux: Optional[Channel] = None
    qubits: Optional[List["Qubit"]] = field(default_factory=dict)

    @property
    def channels(self):
        if self.flux is not None:
            yield self.flux
