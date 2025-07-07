import json
from typing import Any, Optional

from qblox_instruments import SequencerStatus, SequencerStatuses

__all__ = []


class MockSequencer:
    def __init__(self, idx: int, ancestors: list) -> None:
        self.idx = idx
        self.register = {"calls": []}
        self.ancestors = [self] + ancestors

    @property
    def seq_idx(self):
        return self.idx

    def __getattribute__(self, name: str):
        if name in ["idx", "register", "ancestors", "seq_idx"]:
            return super().__getattribute__(name)

        log: dict["str", Any] = {"name": name}
        self.register["calls"].append(log)

        def wrapped(*args, **kwargs):
            log["args"], log["kwargs"] = args, kwargs

        return wrapped


class MockModule:
    def __init__(self, slot: int) -> None:
        self.slot_idx = slot
        self.register = {"calls": []}
        self.sequencers = [MockSequencer(i, [self]) for i in range(6)]

    def present(self) -> bool:
        return True

    @property
    def is_rf_type(self) -> bool:
        return True

    @property
    def is_qrm_type(self) -> bool:
        return True

    def snapshot(self) -> dict:
        return self.register | {
            "sequencers": {seq.idx: seq.register for seq in self.sequencers}
        }

    def __getattribute__(self, name: str):
        if name in [
            "slot_idx",
            "register",
            "sequencers",
            "present",
            "is_rf_type",
            "is_qrm_type",
            "snapshot",
        ]:
            return super().__getattribute__(name)

        log: dict["str", Any] = {"name": name}
        self.register["calls"].append(log)

        def wrapped(*args, **kwargs):
            log["args"], log["kwargs"] = args, kwargs

        return wrapped


class MockCluster:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.resets: int = 0
        self.modules = [MockModule(slot) for slot in range(21)]

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        pass

    @property
    def records(self) -> dict:
        return {
            "args": self.args,
            "kwargs": self.kwargs,
            "resets": self.resets,
            "modules": {mod.slot_idx: mod.snapshot() for mod in self.modules},
        }

    def snapshot(self) -> str:
        return json.dumps(self.records)

    @property
    def sequences(self) -> dict[tuple[int, int], Optional[dict]]:
        return {
            k: v
            for k, v in {
                (mod.slot_idx, seq.idx): next(
                    iter(
                        call["args"][1]
                        for call in seq.register["calls"]
                        if call["name"] == "set" and call["args"][0] == "sequence"
                    ),
                    None,
                )
                for mod in self.modules
                for seq in mod.sequencers
            }.items()
            if v is not None
        }

    @property
    def programs(self) -> dict[tuple[int, int], Optional[dict]]:
        return {
            id_: seq["program"] if seq is not None else None
            for id_, seq in self.sequences.items()
        }

    def get_sequencer_status(
        self, slot: int, sequencer: int, *, timeout: int
    ) -> SequencerStatus:
        return SequencerStatus(
            status=SequencerStatuses.OKAY,
            state=None,
            info_flags=None,
            warn_flags=None,
            err_flags=None,
            log=None,
        )

    def get_acquisition_status(self, slot: int, sequencer: int, *, timeout: int):
        pass

    def get_acquisitions(self, slot: int, sequencer: int) -> dict:
        return {}

    def reference_source(self, value=None) -> None:
        pass
