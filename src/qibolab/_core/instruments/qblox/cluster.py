from qibolab._core.components.configs import Config
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import Model
from qibolab._core.sweeper import ParallelSweepers

from .sequence import Sequence

__all__ = ["Cluster"]

SAMPLING_RATE = 1


class PortAddress(Model):
    module: int
    port: int
    input: bool = False

    @classmethod
    def from_path(cls, path: str):
        """Load address from :attr:`qibolab.Channel.path`."""
        els = path.split("/")
        assert len(els) == 2
        return cls(module=int(els[0]), port=int(els[1][1:]), input=els[1][0] == "i")


class Cluster(Controller):
    name: str
    """Device name.

    As described in:
    https://docs.qblox.com/en/main/getting_started/setup.html#connecting-to-multiple-instruments
    """
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
        results = {}
        for ps in sequences:
            seq = Sequence.from_pulses(ps, sweepers, options)
            results |= self._execute([seq])
        return results

    def _execute(self, sequences: list[Sequence]) -> dict:
        return {}
