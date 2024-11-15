from typing import Optional

import qblox_instruments as qblox
from qblox_instruments.qcodes_drivers.module import Module
from qcodes.instrument import find_or_create_instrument

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
    _cluster: Optional[qblox.Cluster] = None

    @property
    def sampling_rate(self) -> int:
        return SAMPLING_RATE

    def connect(self):
        self._cluster = find_or_create_instrument(
            qblox.Cluster, recreate=True, name=self.name, identifier=self.address
        )

    @property
    def is_connected(self) -> bool:
        return self._cluster is not None

    def disconnect(self):
        assert self._cluster is not None

        for module in self.modules.values():
            module.stop_sequencer()
        self._cluster.reset()
        self._cluster = None

    @property
    def modules(self) -> dict[int, Module]:
        assert self._cluster is not None
        return {mod.slot_idx: mod for mod in self._cluster.modules if mod.present()}

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
