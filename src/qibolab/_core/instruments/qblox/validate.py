from qibolab._core.execution_parameters import (
    AcquisitionType,
    ExecutionParameters,
    default,
)
from qibolab._core.identifier import ChannelId
from qibolab._core.pulses import Acquisition as AcquisitionPulse
from qibolab._core.pulses import Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .limits import (
    ACQUISITION_MEMORY,
    ACQUISITION_NUMBER,
    QCM_INSTRUCTION_MEMORY,
    QRM_INSTRUCTION_MEMORY,
    SCOPE_MEMORY,
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
    sequence: Q1Sequence, acquisition_type: AcquisitionType
) -> None:
    """Ensure each sequencer performs no more than one RAW acquisition per run.

    A run may perform multiple scope acquisitions if they are assigned to
    different sequencers. Because a sequencer cannot acquire more than once per
    run, RAW sequences are not batched; batching could place multiple
    acquisitions on the same sequencer and violate this constraint.
    """
    if acquisition_type is not AcquisitionType.RAW:
        return

    n_acquisitions = len(sequence.acquisitions)
    if n_acquisitions > 1:
        raise ValueError(
            "RAW acquisition supports at most one acquisition pulse per "
            f"sequencer: {n_acquisitions}."
        )


def validate_sequence(
    sequence: Q1Sequence, is_qrm: bool, acquisition_type: AcquisitionType
) -> None:
    """Validate sequence memory limits and RAW acquisition constraints."""
    assert_instruction_memory(sequence.program.lines, is_qrm)
    assert_acquisition_memory(list(sequence.acquisitions.values()))
    assert_waveform_memory(list(sequence.waveforms.values()))
    assert_weight_memory(list(sequence.weights.values()))
    validate_raw_acquisitions(sequence, acquisition_type)


def validate_raw_options(
    options: ExecutionParameters, sweepers: list[ParallelSweepers]
) -> None:
    """Reject RAW sweepers and averaging modes that exceed Qblox scope limits.

    RAW acquisition supports neither sweepers nor single-shot averaging: the scope
    memory cannot retain a separate trace for every shot or sweep point. Any averaged
    mode is accepted, since the Qblox driver executes ``SEQUENTIAL`` and ``CYCLIC``
    identically (a single hardware-averaged trace).
    """
    if options.acquisition_type is not AcquisitionType.RAW:
        return

    if any(sweepers):
        raise ValueError("RAW acquisition does not support sweepers.")

    if not options.averaging_mode.average:
        # Qblox scope memory cannot retain a separate RAW trace for every shot.
        raise NotImplementedError(
            "Qblox RAW acquisition requires an averaged AveragingMode because of "
            "scope acquisition memory limits; AveragingMode.SINGLESHOT is not "
            "supported."
        )


def validate_raw_duration(
    sequences: list[PulseSequence], options: ExecutionParameters
) -> None:
    """Require each RAW shot to span the whole scope memory.

    The scope capture restarts at every shot. Samples never written by the current
    experiment retain data from previous ones, since hardware averaging does not reset
    them. Acquisition and relaxation are a lower bound of the time between two captures,
    so together they must fill the trace.
    """
    if options.acquisition_type is not AcquisitionType.RAW:
        return

    relaxation = default(options.relaxation_time, 0)
    for sequence in sequences:
        for _, event in sequence:
            acq = event.acquisition if isinstance(event, Readout) else event
            if (
                isinstance(acq, AcquisitionPulse)
                and acq.duration + relaxation < SCOPE_MEMORY
            ):
                raise ValueError(
                    f"RAW acquisition requires acquisition duration ({acq.duration:g} "
                    f"ns) plus relaxation time ({relaxation:g} ns) to be at least "
                    f"{SCOPE_MEMORY} ns, to fill the scope memory."
                )
