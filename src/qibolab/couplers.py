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

    def __post_init__(self):
        # register qubit in ``flux`` channel so that we can access
        # ``sweetspot`` and ``filters`` at the channel level
        if self.flux:
            self.flux.coupler = self

    @property
    def channels(self):
        for channel in [self.flux]:
            if channel is not None:
                yield channel
