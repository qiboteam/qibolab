import json
from typing import Optional, cast

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

from ..sequence import Q1Sequence
from .port import PortAddress

__all__ = []


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
