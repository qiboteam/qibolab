from functools import reduce
from operator import or_

from qblox_instruments.qcodes_drivers.module import Module

from qibolab._core.identifier import ChannelId

from .identifiers import SequencerMap, SlotId
from .sequence import Sequence, acquisition

__all__ = []


def integration_lenghts(
    sequences: dict[ChannelId, Sequence],
    sequencers: SequencerMap,
    modules: dict[SlotId, Module],
) -> dict[acquisition.MeasureId, int]:
    channels_to_sequencer = {
        ch: (mod, seq) for mod, chs in sequencers.items() for ch, seq in chs.items()
    }
    return acquisition.integration_lengths(
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
