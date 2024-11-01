from qibolab._core.components.configs import Config
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

__all__ = []

SAMPLING_RATE = 0


class Cluster(Controller):
    bounds: str = "qblox/bounds"

    def connect(self):
        pass

    def disconnect(self):
        pass

    @property
    def sampling_rate(self) -> int:
        return SAMPLING_RATE

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:
        return {}
