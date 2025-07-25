from qibolab._core.identifier import ChannelId

__all__ = []

SlotId = int
SequencerId = int
SequencerMap = dict[SlotId, dict[ChannelId, SequencerId]]
