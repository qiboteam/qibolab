import json
from typing import Any

__all__ = []


class MockSequencer:
    def __init__(self, idx: int, ancestors: list) -> None:
        self.idx = idx
        self.register = {"calls": []}
        self.ancestors = [self] + ancestors

    def __getattribute__(self, name: str):
        if name in ["idx", "register", "ancestors"]:
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
        self.sequencers = [MockSequencer(i, [self]) for i in range(10)]

    def present(self) -> bool:
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

    def snapshot(self) -> str:
        return json.dumps(
            {
                "args": self.args,
                "kwargs": self.kwargs,
                "resets": self.resets,
                "modules": {mod.slot_idx: mod.snapshot() for mod in self.modules},
            }
        )

    def get_sequencer_status(self, slot: int, sequencer: int) -> str:
        return ""

    def get_acquisition_status(self, slot: int, sequencer: int, *, timeout: int):
        pass

    def get_acquisitions(self, slot: int, sequencer: int) -> dict:
        return {}
