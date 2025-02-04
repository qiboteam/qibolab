from qibolab._core import pulses
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import Model

from .waveforms import Waveform

__all__ = []

MeasureId = str
Weight = Waveform
Weights = dict[str, Weight]


class Acquisition(Model):
    num_bins: int
    index: int


Acquisitions = dict[MeasureId, Acquisition]


class AcquisitionSpec(Model):
    acquisition: Acquisition
    duration: int


def acquisitions(
    sequence: PulseSequence, num_bins: int
) -> dict[MeasureId, AcquisitionSpec]:
    return {
        str(acq.id): AcquisitionSpec(
            acquisition=Acquisition(num_bins=num_bins, index=i),
            duration=int(acq.duration),
        )
        for i, acq in enumerate(
            p for _, p in sequence if isinstance(p, pulses.Acquisition)
        )
    }
