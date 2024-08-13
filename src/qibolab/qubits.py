from typing import Annotated, Optional, Union

from pydantic import ConfigDict, Field

from qibolab.components import AcquireChannel, DcChannel, IqChannel
from qibolab.native import SingleQubitNatives, TwoQubitNatives
from qibolab.serialize_ import Model

QubitId = Annotated[Union[int, str], Field(union_mode="left_to_right")]
"""Type for qubit names."""

CHANNEL_NAMES = ("probe", "acquisition", "drive", "drive12", "drive_cross", "flux")
"""Names of channels that belong to a qubit.

Not all channels are required to operate a qubit.
"""


class Qubit(Model):
    """Representation of a physical qubit.

    Qubit objects are instantiated by :class:`qibolab.platforms.platform.Platform`
    but they are passed to instrument designs in order to play pulses.

    Args:
        name (int, str): Qubit number or name.
        readout (:class:`qibolab.platforms.utils.Channel`): Channel used to
            readout pulses to the qubit.
        drive (:class:`qibolab.platforms.utils.Channel`): Channel used to
            send drive pulses to the qubit.
        flux (:class:`qibolab.platforms.utils.Channel`): Channel used to
            send flux pulses to the qubit.
    """

    model_config = ConfigDict(frozen=False)

    name: QubitId

    native_gates: Annotated[
        SingleQubitNatives, Field(default_factory=SingleQubitNatives)
    ]

    probe: Optional[IqChannel] = None
    acquisition: Optional[AcquireChannel] = None
    drive: Optional[IqChannel] = None
    drive12: Optional[IqChannel] = None
    drive_cross: Optional[dict[QubitId, IqChannel]] = None
    flux: Optional[DcChannel] = None

    @property
    def channels(self):
        for name in CHANNEL_NAMES:
            channel = getattr(self, name)
            if channel is not None:
                yield channel

    @property
    def mixer_frequencies(self):
        """Get local oscillator and intermediate frequencies of native gates.

        Assumes RF = LO + IF.
        """
        freqs = {}
        for name in self.native_gates.model_fields:
            native = getattr(self.native_gates, name)
            if native is not None:
                channel_type = native.pulse_type.name.lower()
                _lo = getattr(self, channel_type).lo_frequency
                _if = native.frequency - _lo
                freqs[name] = _lo, _if
        return freqs


QubitPairId = tuple[QubitId, QubitId]
"""Type for holding ``QubitPair``s in the ``platform.pairs`` dictionary."""


class QubitPair(Model):
    """Data structure for holding the native two-qubit gates acting on a pair
    of qubits.

    This is needed for symmetry to the single-qubit gates which are storred in the
    :class:`qibolab.platforms.abstract.Qubit`.
    """

    qubit1: QubitId
    """First qubit of the pair.

    Acts as control on two-qubit gates.
    """
    qubit2: QubitId
    """Second qubit of the pair.

    Acts as target on two-qubit gates.
    """

    native_gates: Annotated[TwoQubitNatives, Field(default_factory=SingleQubitNatives)]
