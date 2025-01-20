import time
from collections import defaultdict
from functools import cached_property
from itertools import groupby
from typing import Optional

import qblox_instruments as qblox
from qblox_instruments.qcodes_drivers.module import Module
from qcodes.instrument import find_or_create_instrument

from qibolab._core.components.configs import Config
from qibolab._core.execution_parameters import AcquisitionType, ExecutionParameters
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from . import config
from .config import PortAddress, SeqeuencerMap, SlotId
from .log import Logger
from .sequence import Sequence, compile
from .sequence.acquisition import AcquiredData

__all__ = ["Cluster"]

SAMPLING_RATE = 1


class Cluster(Controller):
    name: str
    """Device name.

    As described in:
    https://docs.qblox.com/en/main/getting_started/setup.html#connecting-to-multiple-instruments
    """
    bounds: str = "qblox/bounds"
    _cluster: Optional[qblox.Cluster] = None

    @property
    def cluster(self) -> qblox.Cluster:
        assert self._cluster is not None
        return self._cluster

    @cached_property
    def _modules(self) -> dict[SlotId, Module]:
        return {mod.slot_idx: mod for mod in self.cluster.modules if mod.present()}

    @cached_property
    def _channels_by_module(self) -> dict[SlotId, list[tuple[ChannelId, PortAddress]]]:
        addresses = {
            name: PortAddress.from_path(ch.path) for name, ch in self.channels.items()
        }
        return {
            k: [el[1] for el in g]
            for k, g in groupby(
                sorted(
                    (address.slot, (ch, address)) for ch, address in addresses.items()
                ),
                key=lambda el: el[0],
            )
        }

    @property
    def sampling_rate(self) -> int:
        return SAMPLING_RATE

    def connect(self):
        if self.is_connected:
            return

        self._cluster = find_or_create_instrument(
            qblox.Cluster, recreate=True, name=self.name, identifier=self.address
        )
        self._cluster.reset()

    @property
    def is_connected(self) -> bool:
        return self._cluster is not None

    def disconnect(self):
        assert self._cluster is not None

        for module in self._modules.values():
            module.stop_sequencer()
        self._cluster.reset()
        self._cluster = None

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:
        results = {}
        log = Logger(configs)

        for ps in sequences:
            sequences_ = compile(ps, sweepers, options, self.sampling_rate)
            log.sequences(sequences_)
            sequencers = self._prepare(sequences_, options.acquisition_type)
            log.status(self.cluster, sequencers)
            data = self._execute(sequencers, options.estimate_duration([ps], sweepers))
            log.data(data)
            results |= _extract(data)
        return results

    def _prepare(
        self, sequences: dict[ChannelId, Sequence], acquisition: AcquisitionType
    ) -> SeqeuencerMap:
        sequencers = defaultdict(dict)
        for slot, chs in self._channels_by_module.items():
            module = self._modules[slot]
            assert len(module.sequencers) > len(chs)
            config.module(module)
            for idx, ((ch, address), sequencer) in enumerate(
                zip(chs, module.sequencers)
            ):
                sequencers[slot][ch] = idx
                config.sequencer(
                    sequencer, address, sequences.get(ch, Sequence.empty()), acquisition
                )

        return sequencers

    def _execute(
        self, sequencers: SeqeuencerMap, duration: float
    ) -> dict[ChannelId, AcquiredData]:
        # TODO: implement
        for mod, seqs in sequencers.items():
            module = self._modules[mod]
            for seq in seqs.values():
                module.arm_sequencer(seq)
            module.start_sequencer()

        time.sleep(duration + 1)

        acquisitions = {}
        for slot, seqs in sequencers.items():
            for ch, seq in seqs.items():
                self.cluster.get_acquisition_status(slot, seq, timeout=10)
                acquisitions[ch] = self.cluster.get_acquisitions(slot, seq)

        return acquisitions


def _extract(acquisitions: dict[ChannelId, dict]) -> dict[PulseId, Result]:
    return {}
