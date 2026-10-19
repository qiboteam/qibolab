from qibolab._core.identifier import ChannelId
from qibolab._core.sequence import PulseSequence

from .limits import (
    ACQUISITION_MEMORY,
    ACQUISITION_NUMBER,
    QCM_INSTRUCTION_MEMORY,
    QRM_INSTRUCTION_MEMORY,
    WAVEFORM_MEMORY,
    WAVEFORM_NUMBER,
    WEIGHT_MEMORY,
    WEIGHT_NUMBER,
)
from .q1asm.ast_ import Line
from .sequence.acquisition import Acquisition, Weight
from .sequence.sequence import Q1Sequence
from .sequence.waveforms import Waveform

__all__ = []


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


def assert_instruction_memory(lines: list[Line], is_qrm: bool) -> None:
    """Assert instruction memory limitations.

    The limit depends on the hosting module kind (QRM vs QCM), not on channel type.
    """
    mem = QRM_INSTRUCTION_MEMORY if is_qrm else QCM_INSTRUCTION_MEMORY
    assert len(lines) <= mem


def validate_sequence(sequence: Q1Sequence, is_qrm: bool) -> None:
    """Validate sequence elements."""
    assert_instruction_memory(sequence.program.lines, is_qrm)
    assert_acquisition_memory(list(sequence.acquisitions.values()))
    assert_waveform_memory(list(sequence.waveforms.values()))
    assert_weight_memory(list(sequence.weights.values()))
