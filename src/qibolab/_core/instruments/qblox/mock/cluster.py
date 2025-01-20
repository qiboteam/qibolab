__all__ = []


from ..sequence import Sequence


class MockSequencer:
    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.register = {}

    def sequence(self, sequence: Sequence) -> None:
        pass


class MockModule:
    def __init__(self, slot: int) -> None:
        self.slot_idx = slot

    def present(self) -> bool:
        return True

    @property
    def is_qrm_type(self) -> bool:
        return True

    @property
    def sequencers(self) -> list:
        return [MockSequencer(i) for i in range(20)]

    def disconnect_outputs(self) -> None:
        pass

    def scope_acq_trigger_mode_path0(self, mode: str) -> None:
        pass

    def scope_acq_trigger_mode_path1(self, mode: str) -> None:
        pass


class MockCluster:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.resets: int = 0

    @property
    def modules(self) -> list:
        return [MockModule(slot) for slot in range(21)]

    def reset(self) -> None:
        self.resets += 1
