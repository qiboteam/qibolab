from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import ChannelId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

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


def validate_raw_acquisitions(
    sequence: Q1Sequence,
    parsweepers_list: list[ParallelSweepers],
    acquisition_type: AcquisitionType,
) -> None:
    """Ensure each sequencer performs no more than one RAW acquisition per run.

    A run may perform multiple scope acquisitions if they are assigned to
    different sequencers. Because a sequencer cannot acquire more than once per
    run, RAW sequences are not batched; batching could place multiple
    acquisitions on the same sequencer and violate this constraint.
    """
    if any(len(sweepers) > 0 for sweepers in parsweepers_list):
        raise ValueError("RAW acquisition does not support sweepers.")
    if acquisition_type is AcquisitionType.RAW and len(sequence.acquisitions) > 1:
        raise ValueError(
            "RAW acquisition supports at most one acquisition pulse per "
            f"sequencer: {len(sequence.acquisitions)}."
        )


def validate_sequence(
    sequence: Q1Sequence,
    parsweepers_list: list[ParallelSweepers],
    is_qrm: bool,
    acquisition_type: AcquisitionType,
) -> None:
    """Validate sequence memory limits and RAW acquisition constraints."""
    assert_instruction_memory(sequence.program.lines, is_qrm)
    assert_acquisition_memory(list(sequence.acquisitions.values()))
    assert_waveform_memory(list(sequence.waveforms.values()))
    assert_weight_memory(list(sequence.weights.values()))
    validate_raw_acquisitions(sequence, parsweepers_list, acquisition_type)


def validate_raw_averaging_mode(options: ExecutionParameters) -> None:
    """Reject RAW averaging modes that exceed Qblox scope-memory limits."""
    if (
        options.acquisition_type is AcquisitionType.RAW
        and options.averaging_mode is not AveragingMode.CYCLIC
    ):
        # Qblox scope memory cannot retain a separate RAW trace for every shot.
        raise NotImplementedError(
            "Qblox RAW acquisition requires AveragingMode.CYCLIC because of "
            "scope acquisition memory limits."
        )
