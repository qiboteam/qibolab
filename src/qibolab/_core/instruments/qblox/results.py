from functools import reduce
from operator import or_
from typing import Optional, TypedDict

import numpy as np
from qblox_instruments.qcodes_drivers.module import Module

from qibolab._core.identifier import ChannelId, Result
from qibolab._core.pulses.pulse import PulseId

from .identifiers import SequencerId, SequencerMap, SlotId
from .sequence import Sequence, acquisition

__all__ = []


def _fill_empty_lenghts(
    weighted: dict[acquisition.MeasureId, Optional[int]],
    defaults: dict[tuple[SlotId, SequencerId], int],
    locations: dict[acquisition.MeasureId, tuple[SlotId, SequencerId]],
) -> dict[acquisition.MeasureId, int]:
    return {
        k: v if v is not None else defaults[locations[k]] for k, v in weighted.items()
    }


def integration_lenghts(
    sequences: dict[ChannelId, Sequence],
    sequencers: SequencerMap,
    modules: dict[SlotId, Module],
) -> dict[acquisition.MeasureId, int]:
    channels_to_sequencer = {
        ch: (mod, seq) for mod, chs in sequencers.items() for ch, seq in chs.items()
    }
    return _fill_empty_lenghts(
        reduce(or_, (seq.integration_lengths for seq in sequences.values())),
        {
            (mod_id, seq): seq.integration_length_acq()
            for mod_id, mod in modules.items()
            for seq in mod.sequencers
        },
        {
            acq: channels_to_sequencer[ch]
            for ch, seq in sequences.items()
            for acq in seq.acquisitions
        },
    )


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


AcquiredData = dict[acquisition.MeasureId, IndexedData]


def _extract(data: Integration, length: int) -> Result:
    res = np.array([data["path0"], data["path1"]])
    return np.moveaxis(res, 0, -1) / length


def extract(
    acquisitions: dict[ChannelId, AcquiredData],
    lengths: dict[acquisition.MeasureId, int],
) -> dict[PulseId, Result]:
    # TODO: check if the `lengths` info coincide with the
    # idata["acquisition"]["bins"]["avg_cnt"]
    return {
        int(acq): _extract(idata["acquisition"]["bins"]["integration"], lengths[acq])
        for data in acquisitions.values()
        for acq, idata in data.items()
    }
