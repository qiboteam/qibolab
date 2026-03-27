import qblox_instruments

from qibolab._core.components.channels import AcquisitionChannel, Channel
from qibolab._core.identifier import ChannelId
from qibolab._core.sequence import PulseSequence

from .q1asm.ast_ import Line
from .sequence.acquisition import Acquisition, Weight
from .sequence.sequence import Q1Sequence
from .sequence.waveforms import Waveform

__all__ = []

WAVEFORM_MEMORY = 2**14
"""Maximum waveform memory available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#waveform-memory
"""
WAVEFORM_NUMBER = 2**10
"""Maximum number of waveforms available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#waveform-memory
"""

QCM_INSTRUCTION_MEMORY = 2**14
QRM_INSTRUCTION_MEMORY = 12288
"""Maximum number of instructions per program for the QCM and QRM modules.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#instruction-memory
"""


WEIGHT_MEMORY = 2**14
"""Maximum waveform memory available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#integrator
"""
WEIGHT_NUMBER = 2**5
"""Maximum number of weights available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#weight-memory
"""

if qblox_instruments.__version__ >= "1.0.0":
    ACQUISITION_MEMORY = 3e6
else:
    ACQUISITION_MEMORY = 2**17
"""Maximum acquisition memory available.

..note::
    Here the qblox instruments version is used as a proxy for the check if the cluster
    firmware >= 0.13.0, which due to the dependency requirements is an equivalent
    condition.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html
https://docs.qblox.com/en/main/releases.html#new-features
"""

ACQUISITION_NUMBER = 2**5
"""Maximum number of acquisitions available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#:~:text=Square%20Weight%20Acquisitions
"""

# There are used for the batching in cluster.py
cluster_memory_limits = {
    "acq_memory": ACQUISITION_MEMORY,
    "acq_number": ACQUISITION_NUMBER,
    "qcm_lines": QCM_INSTRUCTION_MEMORY,
    "qrm_lines": QRM_INSTRUCTION_MEMORY,
}


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


def assert_instruction_memory(channel: Channel, lines: list[Line]):
    mem = (
        QRM_INSTRUCTION_MEMORY
        if isinstance(channel, AcquisitionChannel)
        else QCM_INSTRUCTION_MEMORY
    )
    assert len(lines) <= mem


def validate_sequence(channel: Channel, sequence: Q1Sequence) -> None:
    """Validate sequence elements."""
    assert_instruction_memory(channel, sequence.program.lines)
    assert_acquisition_memory(list(sequence.acquisitions.values()))
    assert_waveform_memory(list(sequence.waveforms.values()))
    assert_weight_memory(list(sequence.weights.values()))
