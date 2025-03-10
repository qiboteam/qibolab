import json
from typing import Literal, Optional, cast

import numpy as np
from qblox_instruments.qcodes_drivers.module import Module
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qibolab._core.components.channels import Channel, IqChannel
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
    def direction(self) -> Literal["io", "out"]:
        """Signal flow direction.

        This is used for :meth:`local_address`, check its description for the in-depth
        meaning.

        .. note::

            While three directions are actually possible, there is no usage of pure
            acquisition channels in this driver (corresponding to the ``"in""``
            direction), and only support :class:`qibolab.Readout` instructions.

            For this reason, an input channel is always resolved to ``"io"``, implying
            that a same sequencer will be used to control both input (acquisitions) and
            output (probe pulses).
        """
        return "io" if self.input else "out"

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
        return f"{self.direction}{channels}"


def module(
    mod: Module,
    channels: dict[ChannelId, Channel],
    los: dict[ChannelId, OscillatorConfig],
):
    """Configure module-wide settings."""
    # map sequencers to specific outputs (but first disable all sequencer connections)
    mod.disconnect_outputs()

    if mod.is_qrm_type:
        mod.disconnect_inputs()
        # we do not currently support acquisition on external digital trigger
        mod.scope_acq_trigger_mode_path0("sequencer")
        mod.scope_acq_trigger_mode_path1("sequencer")

    # set lo frequencies
    for iq, lo in los.items():
        n = PortAddress.from_path(channels[iq].path).ports[0] - 1
        path = f"out{n}_in{n}" if mod.is_qrm_type else f"out{n}"
        getattr(mod, f"{path}_lo_en")(True)
        getattr(mod, f"{path}_lo_freq")(int(lo.frequency))
        getattr(mod, f"out{n}_att")(int(lo.power))

    if mod.is_qrm_type:
        mod.in0_att(0)


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
    config = configs[channel_id]

    # set parameters
    # offsets
    if isinstance(config, DcConfig):
        seq.ancestors[1].set(f"out{seq.seq_idx}_offset", config.offset)

    # avoid sequence operations for inactive sequencers, including synchronization
    if sequence.is_empty:
        return

    # connect to physical address
    seq.connect_sequencer(address.local_address)

    seq.offset_awg_path0(0.0)
    seq.offset_awg_path1(0.0)
    # modulation, only disable for QCM - always used for flux pulses
    mod = cast(Module, seq.ancestors[1])
    seq.mod_en_awg(mod.is_rf_type)

    # FIX: for no apparent reason other than experimental evidence, the marker has to be
    # enabled and set to a certain value
    seq.marker_ovr_en(True)
    seq.marker_ovr_value(15)

    # acquisition
    if address.input:
        assert isinstance(config, AcquisitionConfig)
        length = _integration_length(sequence)
        if length is not None:
            seq.integration_length_acq(length)
        # discrimination
        if config.iq_angle is not None:
            seq.thresholded_acq_rotation(np.degrees(config.iq_angle % (2 * np.pi)))
        if config.threshold is not None:
            seq.thresholded_acq_threshold(config.threshold)
        # demodulation
        seq.demod_en_acq(acquisition is not AcquisitionType.RAW)

    probe = channels[channel_id].iqout(channel_id)
    if probe is not None:
        freq = cast(IqConfig, configs[probe]).frequency
        lo = cast(IqChannel, channels[probe]).lo
        assert lo is not None
        lo_freq = cast(OscillatorConfig, configs[lo]).frequency
        seq.nco_freq(int(freq - lo_freq))

    # upload sequence
    # - ensure JSON compatibility of the sent dictionary
    seq.sequence(json.loads(sequence.model_dump_json()))

    # configure the sequencers to synchronize
    seq.sync_en(True)
