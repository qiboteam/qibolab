import json
from typing import Optional, cast

import numpy as np
from qblox_instruments.qcodes_drivers.module import Module
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qibolab._core.components.channels import AcquisitionChannel, Channel, IqChannel
from qibolab._core.components.configs import (
    AcquisitionConfig,
    Configs,
    DcConfig,
    IqConfig,
    OscillatorConfig,
)
from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.identifier import ChannelId
from qibolab._core.serialize import Model

from .identifiers import SlotId
from .sequence import Q1Sequence

__all__ = []


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
        for port in self.ports:
            if port is not None:
                assert port > 0
        channels = (
            str(self.ports[0] - 1)
            if self.ports[1] is None
            else f"{self.ports[0] - 1}_{self.ports[1] - 1}"
        )
        return f"out{channels}"


def _iqout(id_: ChannelId, channel: Channel) -> Optional[ChannelId]:
    """Extract associated IQ output channel.

    This is the identity for each IQ output channel identifier, while it retrieves the
    associated probe channel for acquisition ones, and :obj:`None` for any other one
    (essentially, non-RF channels).
    """
    return (
        id_
        if isinstance(channel, IqChannel)
        else (
            channel.probe
            if isinstance(channel, AcquisitionChannel) and channel.probe is not None
            else None
        )
    )


def _los(
    mod_channels: set[ChannelId], channels: dict[ChannelId, Channel]
) -> set[tuple[ChannelId, str]]:
    """Extract all LOs of a certain module.

    The result contains the associated channel, since required to
    address the LO through the API. While the LO identifier is used to
    retrieve the configuration.
    """
    return {
        (iq, lo)
        for iq, lo in (
            (iq, cast(IqChannel, channels[iq]).lo)
            for iq in (_iqout(ch, channels[ch]) for ch in mod_channels)
            if iq is not None
        )
        if lo is not None
    }


def module(
    mod: Module,
    mod_channels: set[ChannelId],
    channels: dict[ChannelId, Channel],
    configs: Configs,
):
    """Configure module-wide settings."""
    # map sequencers to specific outputs (but first disable all sequencer connections)
    mod.disconnect_outputs()

    if mod.is_qrm_type:
        # we do not currently support acquisition on external digital trigger
        mod.scope_acq_trigger_mode_path0("sequencer")
        mod.scope_acq_trigger_mode_path1("sequencer")

    # set lo frequencies
    for iq, lo in _los(mod_channels, channels):
        n = PortAddress.from_path(channels[iq].path).ports[0] - 1
        attr = f"out{n}_in{n}_lo_freq" if mod.is_qrm_type else f"out{n}_lo_freq"
        getattr(mod, attr)(cast(OscillatorConfig, configs[lo]).frequency)


def _integration_length(sequence: Q1Sequence) -> Optional[int]:
    """Find integration length based on sequence waveform lengths."""
    lengths = {len(waveform.data) for waveform in sequence.waveforms.values()}
    if len(lengths) == 0:
        return None
    if len(lengths) == 1:
        return lengths.pop()
    raise NotImplementedError(
        "Cannot acquire different lengths using the same sequencer."
    )


def sequencer(
    seq: Sequencer,
    address: PortAddress,
    sequence: Q1Sequence,
    channel_id: ChannelId,
    channels: dict[ChannelId, Channel],
    configs: Configs,
    acquisition: AcquisitionType,
):
    """Configure sequencer-wide settings."""
    # upload sequence
    # - ensure JSON compatibility of the sent dictionary
    seq.sequence(json.loads(sequence.model_dump_json()))

    # configure the sequencers to synchronize
    seq.sync_en(True)

    config = configs[channel_id]

    # set parameters
    # offsets
    seq.offset_awg_path0(config.offset if isinstance(config, DcConfig) else 0.0)
    seq.offset_awg_path1(0.0)
    # modulation, only disable for QCM - always used for flux pulses
    mod = cast(Module, seq.ancestors[1])
    seq.mod_en_awg(mod.is_qrm_type or mod.is_rf_type)
    # acquisition
    if address.input:
        assert isinstance(config, AcquisitionConfig)
        length = _integration_length(sequence)
        if length is not None:
            seq.integration_length_acq(length)
        # discrimination
        seq.thresholded_acq_rotation(np.degrees(config.iq_angle % (2 * np.pi)))
        seq.thresholded_acq_threshold(config.threshold)
        # demodulation
        seq.demod_en_acq(acquisition is not AcquisitionType.RAW)

    probe = _iqout(channel_id, channels[channel_id])
    if probe is not None:
        freq = cast(IqConfig, configs[probe]).frequency
        lo = cast(IqChannel, channels[probe]).lo
        assert lo is not None
        lo_freq = cast(OscillatorConfig, configs[lo]).frequency
        seq.nco_freq(freq - lo_freq)

    # connect to physical address
    seq.connect_sequencer(address.local_address)
