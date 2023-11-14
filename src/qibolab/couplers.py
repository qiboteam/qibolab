from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from qibolab.channels import Channel
from qibolab.native import CouplerNatives

QubitId = Union[str, int]
"""Type for Coupler names."""


@dataclass
class Coupler:
    """Representation of a physical coupler.

    Coupler objects are instantiated by :class:`qibolab.platforms.platform.Platform`
    but they are passed to instrument designs in order to play pulses.
    """

    name: QubitId
    "Coupler number or name."

    sweetspot: float = 0
    "Coupler sweetspot to center it's flux dependence if needed."
    native_pulse: CouplerNatives = field(default_factory=CouplerNatives)
    "For now this only contains the calibrated pulse to activate the coupler."

    flux: Optional[Channel] = None
    "flux (:class:`qibolab.platforms.utils.Channel`): Channel used to send flux pulses to the qubit."

    # TODO: With topology or conectivity
    # qubits: Optional[Dict[QubitId, Qubit]] = field(default_factory=dict)
    qubits: Dict = field(default_factory=dict)
    "Qubits the coupler acts on"

    @property
    def channels(self):
        if self.flux is not None:
            yield self.flux
