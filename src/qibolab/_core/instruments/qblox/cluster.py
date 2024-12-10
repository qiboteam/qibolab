from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Optional

import qblox_instruments as qblox
from qblox_instruments.qcodes_drivers.module import Module

from qibolab._core.components.configs import Config
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import Model
from qibolab._core.sweeper import ParallelSweepers

from .sequence import Sequence

# from qcodes.instrument import find_or_create_instrument


__all__ = ["Cluster"]

SAMPLING_RATE = 1


SlotId = int
SequencerId = int


class PortAddress(Model):
    slot: SlotId
    ports: tuple[int, Optional[int]]
    input: bool = False

    @classmethod
    def from_path(cls, path: str):
        """Load address from :attr:`qibolab.Channel.path`."""
        els = path.split("/")
        assert len(els) == 2
        ports = els[1][1:].split("_")
        assert 1 <= len(ports) <= 2
        return cls(
            slot=int(els[0]),
            ports=(int(ports[0]), int(ports[1]) if len(ports) == 2 else None),
            input=els[1][0] == "i",
        )

    @property
    def local_address(self):
        """Physical address within the module.

        It will generate a string in the format ``<direction><channel>`` or
        ``<direction><I-channel>_<Q-channel>``.
        ``<direction>`` is ``in`` for a connection between an input and the acquisition
        path, ``out`` for a connection from the waveform generator to an output, or
        ``io`` to do both.
        The channels must be integer channel indices.
        Only one channel is present for a real mode operating sequencer; two channels
        are used for complex mode.

        .. note::

            Description adapted from
            https://docs.qblox.com/en/main/api_reference/cluster.html#qblox_instruments.Cluster.connect_sequencer
        """
        direction = "in" if self.input else "out"
        channels = (
            str(self.ports[0])
            if self.ports[1] is None
            else f"{self.ports[0]}_{self.ports[1]}"
        )
        return f"{direction}{channels}"


class Cluster(Controller):
    name: str
    """Device name.

    As described in:
    https://docs.qblox.com/en/main/getting_started/setup.html#connecting-to-multiple-instruments
    """
    bounds: str = "qblox/bounds"
    _cluster: Optional[qblox.Cluster] = None

    @cached_property
    def _modules(self) -> dict[SlotId, Module]:
        assert self._cluster is not None
        return {mod.slot_idx: mod for mod in self._cluster.modules if mod.present()}

    @property
    def sampling_rate(self) -> int:
        return SAMPLING_RATE

    def connect(self):
        if self.is_connected:
            return

        # self._cluster = find_or_create_instrument(
        #     qblox.Cluster, recreate=True, name=self.name, identifier=self.address
        # )
        # self._cluster.reset()

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
        for ps in sequences:
            sequences_ = _prepare(ps, sweepers, options, self.sampling_rate)
            if "log" in configs:
                _dump_sequences(configs["log"].path, sequences_)
            sequencers = self._upload(sequences_)
            results |= self._execute(sequencers)
        return results

    def _upload(
        self, sequences: dict[ChannelId, Sequence]
    ) -> dict[SlotId, dict[ChannelId, SequencerId]]:
        sequencers = defaultdict(dict)
        for mod, chs in _channels_by_module().items():
            module = self._modules[mod]
            assert len(module.sequencers) > len(chs)
            # Map sequencers to specific outputs (but first disable all sequencer connections)
            module.disconnect_outputs()
            for idx, (ch, sequencer) in enumerate(zip(chs, module.sequencers)):
                sequencers[mod][ch] = idx
                sequencer.sequence(sequences[ch].model_dump())
                # Configure the sequencers to synchronize
                sequencer.sync_en(True)
                sequencer.connect_sequencer(
                    PortAddress.from_path(self.channels[ch].path).local_address
                )

        return sequencers

    def _execute(self, sequencers: dict[SlotId, dict[ChannelId, SequencerId]]) -> dict:
        # TODO: implement
        for mod, seqs in sequencers.items():
            module = self._modules[mod]
            for seq in seqs.values():
                module.arm_sequencer(seq)
            module.start_sequencer()

        return {}


def _channels_by_module() -> dict[SlotId, list[ChannelId]]:
    # TODO: implement
    return {}


def _prepare(
    sequence: PulseSequence,
    sweepers: list[ParallelSweepers],
    options: ExecutionParameters,
    sampling_rate: float,
) -> dict[ChannelId, Sequence]:
    def ch_pulses(channel: ChannelId):
        return PulseSequence((ch, pulse) for ch, pulse in sequence if ch == channel)

    return {
        channel: Sequence.from_pulses(
            ch_pulses(channel), sweepers, options, sampling_rate
        )
        for channel in sequence.channels
    }


def _dump_sequences(log: Path, sequences: dict[ChannelId, Sequence]):
    for ch, seq in sequences.items():
        (log / f"{ch}.json".replace("/", "-")).write_text(seq.model_dump_json())
