from typing import Optional

from qblox_instruments.qcodes_drivers.module import Module
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qibolab._core.identifier import ChannelId
from qibolab._core.serialize import Model

from .sequence import Sequence

SequencerId = int
SlotId = int
SeqeuencerMap = dict[SlotId, dict[ChannelId, SequencerId]]


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


def module(mod: Module):
    # Map sequencers to specific outputs (but first disable all sequencer connections)
    mod.disconnect_outputs()


def sequencer(seq: Sequencer, address: PortAddress, sequence: Sequence):
    seq.sequence(sequence.model_dump())
    # Configure the sequencers to synchronize
    seq.sync_en(True)
    seq.connect_sequencer(address.local_address)
