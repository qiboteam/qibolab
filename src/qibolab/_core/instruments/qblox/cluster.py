from collections import defaultdict
from functools import cached_property
from itertools import groupby
from pathlib import Path
from typing import Optional

import qblox_instruments as qblox
from qblox_instruments.qcodes_drivers.module import Module

from qibolab._core.components.channels import Channel
from qibolab._core.components.configs import Config, LogConfig
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import PulseId, PulseLike, Readout
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
                assert isinstance(configs["log"], LogConfig)
                _dump_sequences(configs["log"].path, sequences_)
            sequencers = self._upload(sequences_)
            results |= self._execute(sequencers)
        return results

    def _upload(
        self, sequences: dict[ChannelId, Sequence]
    ) -> dict[SlotId, dict[ChannelId, SequencerId]]:
        sequencers = defaultdict(dict)
        for mod, chs in _channels_by_module(self.channels).items():
            module = self._modules[mod]
            assert len(module.sequencers) > len(chs)
            # Map sequencers to specific outputs (but first disable all sequencer connections)
            module.disconnect_outputs()
            for idx, ((ch, address), sequencer) in enumerate(
                zip(chs, module.sequencers)
            ):
                sequencers[mod][ch] = idx
                sequencer.sequence(sequences[ch].model_dump())
                # Configure the sequencers to synchronize
                sequencer.sync_en(True)
                sequencer.connect_sequencer(address.local_address)

        return sequencers

    def _execute(
        self, sequencers: dict[SlotId, dict[ChannelId, SequencerId]]
    ) -> dict[PulseId, Result]:
        # TODO: implement
        for mod, seqs in sequencers.items():
            module = self._modules[mod]
            for seq in seqs.values():
                module.arm_sequencer(seq)
            module.start_sequencer()

        return {}


def _channels_by_module(
    channels: dict[ChannelId, Channel],
) -> dict[SlotId, list[tuple[ChannelId, PortAddress]]]:
    addresses = {name: PortAddress.from_path(ch.path) for name, ch in channels.items()}
    return {
        k: [el[1] for el in g]
        for k, g in groupby(
            sorted((address.slot, (ch, address)) for ch, address in addresses.items()),
            key=lambda el: el[0],
        )
    }


def _split_channels(sequence: PulseSequence) -> dict[ChannelId, PulseSequence]:
    def unwrap(pulse: PulseLike, output: bool) -> PulseLike:
        return (
            pulse
            if not isinstance(pulse, Readout)
            else pulse.probe if output else pulse.acquisition
        )

    def unwrap_seq(seq: PulseSequence, output: bool) -> PulseSequence:
        return PulseSequence((ch, unwrap(p, output)) for ch, p in seq)

    def ch_pulses(channel: ChannelId) -> PulseSequence:
        return PulseSequence((ch, pulse) for ch, pulse in sequence if ch == channel)

    def probe(channel: ChannelId) -> ChannelId:
        return channel.split("/")[0] + "/probe"

    def split(channel: ChannelId) -> dict[ChannelId, PulseSequence]:
        seq = ch_pulses(channel)
        readouts = any(isinstance(p, Readout) for _, p in seq)
        assert not readouts or probe(channel) not in sequence.channels
        return (
            {channel: seq}
            if not readouts
            else {
                channel: unwrap_seq(seq, output=False),
                probe(channel): unwrap_seq(seq, output=True),
            }
        )

    return {
        ch: seq for channel in sequence.channels for ch, seq in split(channel).items()
    }


def _prepare(
    sequence: PulseSequence,
    sweepers: list[ParallelSweepers],
    options: ExecutionParameters,
    sampling_rate: float,
) -> dict[ChannelId, Sequence]:
    return {
        ch: Sequence.from_pulses(seq, sweepers, options, sampling_rate)
        for ch, seq in _split_channels(sequence).items()
    }


def _dump_sequences(log: Path, sequences: dict[ChannelId, Sequence]):
    for ch, seq in sequences.items():
        (log / f"{ch}.json".replace("/", "-")).write_text(seq.model_dump_json())
