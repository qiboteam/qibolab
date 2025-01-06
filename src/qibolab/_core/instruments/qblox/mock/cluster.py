__all__ = []


class MockCluster:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.resets: int = 0

    @property
    def modules(self):
        return []

    def reset(self):
        self.resets += 1
