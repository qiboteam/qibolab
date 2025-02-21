from collections.abc import Iterable

from pydantic import UUID4

from qibolab._core import pulses
from qibolab._core.pulses.pulse import PulseLike
from qibolab._core.serialize import Model

from .waveforms import Waveform

__all__ = []

MeasureId = UUID4
Weight = Waveform
Weights = dict[MeasureId, Weight]


class Acquisition(Model):
    num_bins: int
    index: int


Acquisitions = dict[MeasureId, Acquisition]


class AcquisitionSpec(Model):
    acquisition: Acquisition
    duration: int


def acquisitions(
    sequence: Iterable[PulseLike], num_bins: int
) -> dict[MeasureId, AcquisitionSpec]:
    return {
        acq.id: AcquisitionSpec(
            acquisition=Acquisition(num_bins=num_bins, index=i),
            duration=int(acq.duration),
        )
        for i, acq in enumerate(
            p for p in sequence if isinstance(p, (pulses.Acquisition, pulses.Readout))
        )
    }
