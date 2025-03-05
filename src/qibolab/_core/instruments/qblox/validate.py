from qibolab._core.identifier import ChannelId
from qibolab._core.instruments.qblox.sequence.acquisition import Acquisition, Weight
from qibolab._core.instruments.qblox.sequence.sequence import Q1Sequence
from qibolab._core.sequence import PulseSequence

from .sequence.waveforms import Waveform

__all__ = []

WAVEFORM_MEMORY = 2**14
"""Maximum waveform memory available.

https://docs.qblox.com/en/main/cluster/q1_sequence_processor.html#waveform-memory
"""
WAVEFORM_NUMBER = 2**10
"""Maximum number of waveforms available.

https://docs.qblox.com/en/main/cluster/q1_sequence_processor.html#waveform-memory
"""

WEIGHT_MEMORY = 2**14
"""Maximum waveform memory available.

https://docs.qblox.com/en/main/cluster/q1_sequence_processor.html#integrator
"""
WEIGHT_NUMBER = 2**5
"""Maximum number of weights available.

https://docs.qblox.com/en/main/cluster/q1_sequence_processor.html#weight-memory
"""

ACQUISITION_MEMORY = 2**24 - 4
"""Maximum acquisition memory available.

https://docs.qblox.com/en/main/cluster/q1_sequence_processor.html#integrator
"""

ACQUISITION_NUMBER = 2**5
"""Maximum number of acquisitions available.

https://docs.qblox.com/en/main/cluster/q1_sequence_processor.html#:~:text=Square%20Weight%20Acquisitions
"""


def assert_channels_exclusion(ps: PulseSequence, excluded: set[ChannelId]) -> None:
    """Deny group of channels.

    Mainly used to deny probe channels, since probe events are currently required to be
    bundled with acquisitions, cf. :class:`qibolab.Readout`.

    An extended discussion about the probe-acquisition merge is found at:
    https://github.com/qiboteam/qibolab/pull/1088#issuecomment-2637800857
    """
    assert not any(ch in excluded for ch, _ in ps), (
        "Probe channels can not be controlled independently, please use Readout events."
    )


def assert_waveform_memory(waveforms: list[Waveform]) -> None:
    """Assert waveform memory limitations."""
    assert len(waveforms) <= WAVEFORM_NUMBER
    assert sum(len(w.data) for w in waveforms) <= WAVEFORM_MEMORY


def assert_weight_memory(weights: list[Weight]) -> None:
    """Assert weight memory limitations."""
    assert len(weights) <= WEIGHT_NUMBER
    assert sum(len(w.data) for w in weights) <= WEIGHT_MEMORY


def assert_acquisition_memory(acquisitions: list[Acquisition]) -> None:
    """Assert acquisition memory limitations."""
    assert len(acquisitions) <= ACQUISITION_NUMBER
    assert sum(a.num_bins for a in acquisitions) <= ACQUISITION_MEMORY


def validate_sequence(sequence: Q1Sequence) -> None:
    """Validate sequence elements."""
    assert_acquisition_memory(list(sequence.acquisitions.values()))
    assert_waveform_memory(list(sequence.waveforms.values()))
    assert_weight_memory(list(sequence.weights.values()))
