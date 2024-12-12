from qibolab._core import pulses
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import Model


class Acquisition(Model):
    num_bins: int
    index: int


Acquisitions = dict[str, Acquisition]


def acquisitions(sequence: PulseSequence, num_bins: int) -> Acquisitions:
    return {
        str(acq.id): Acquisition(num_bins=num_bins, index=i)
        for i, acq in enumerate(
            [p for _, p in sequence if isinstance(p, pulses.Acquisition)]
        )
    }
