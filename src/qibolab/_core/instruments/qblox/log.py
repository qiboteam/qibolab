from pathlib import Path
from typing import Any, Optional

from qblox_instruments import Cluster

from qibolab._core.components.configs import Configs, LogConfig
from qibolab._core.identifier import ChannelId

from .config import SeqeuencerMap
from .sequence import Sequence
from .sequence.acquisition import AcquiredData

__all__ = []


def _check(configs: Configs) -> Optional[Path]:
    if "log" in configs:
        assert isinstance(configs["log"], LogConfig)
        return configs["log"].path


def _sanitize(name: str) -> str:
    return name.replace("/", "-")


class Logger:
    def __init__(self, configs: Configs) -> None:
        self.path = _check(configs)

    def __getattribute__(self, name: str, /) -> Any:
        if super().__getattribute__("path") is None:
            return lambda *args, **kwargs: None
        return super().__getattribute__(name)

    def sequences(self, sequences: dict[ChannelId, Sequence]):
        assert self.path is not None
        for ch, seq in sequences.items():
            (self.path / _sanitize(f"{ch}.json")).write_text(seq.model_dump_json())

    def status(self, cluster: Cluster, sequencers: SeqeuencerMap):
        assert self.path is not None
        status = self.path / "status"
        status.mkdir(exist_ok=True)

        (status / "cluster.json").write_text(str(cluster.snapshot()))
        for slot, seqs in sequencers.items():
            for ch, seq_idx in seqs.items():
                (status / _sanitize(f"{ch}.log")).write_text(
                    str(cluster.get_sequencer_status(slot, seq_idx))
                )

    def data(self, data: dict[ChannelId, AcquiredData]):
        pass
