from typing import TypedDict

from qibolab._core import pulses
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.pulses.pulse import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import Model

__all__ = []

MeasureId = str


class Acquisition(Model):
    num_bins: int
    index: int


Acquisitions = dict[MeasureId, Acquisition]


def acquisitions(sequence: PulseSequence, num_bins: int) -> Acquisitions:
    return {
        str(acq.id): Acquisition(num_bins=num_bins, index=i)
        for i, acq in enumerate(
            p for _, p in sequence if isinstance(p, pulses.Acquisition)
        )
    }


IndividualScope = TypedDict(
    "IndividualScope",
    {"data": list[float], "out-of-range": list[bool], "avg_cnt": list[bool]},
)


class ScopeData(TypedDict):
    path0: IndividualScope
    path1: IndividualScope


class Integration(TypedDict):
    path0: list[float]
    path1: list[float]


class BinData(TypedDict):
    integration: Integration
    threshold: list[int]
    valid: list
    avg_cnt: list[int]


class Data(TypedDict):
    scope: ScopeData
    bins: BinData


class IndexedData(TypedDict):
    index: int
    acquisition: Data


AcquiredData = dict[MeasureId, IndexedData]


def extract(acquisitions: dict[ChannelId, AcquiredData]) -> dict[PulseId, Result]:
    return {}
