import json
from typing import cast

import numpy as np
from pydantic import ConfigDict
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qibolab._core.components.channels import Channel, IqChannel
from qibolab._core.components.configs import (
    AcquisitionConfig,
    Configs,
    IqConfig,
    IqMixerConfig,
    OscillatorConfig,
)
from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.identifier import ChannelId
from qibolab._core.serialize import Model

from ..q1asm.ast_ import Acquire, Line
from ..sequence import Q1Sequence
from .port import PortAddress

__all__ = []


def _integration_length(sequence: Q1Sequence) -> int | None:
    """Find integration length based on sequence waveform lengths."""
    lengths = {
        line.instruction.duration
        for line in sequence.program.elements
        if isinstance(line, Line)
        if isinstance(line.instruction, Acquire)
    }
    if len(lengths) == 0:
        return None
    if len(lengths) == 1:
        return lengths.pop()
    raise NotImplementedError(
        "Cannot acquire different lengths using the same sequencer."
    )


class SequencerConfig(Model):
    # disable freeze, to be able to construct instance with optional fields, but also
    # static validation
    model_config = ConfigDict(frozen=False)

    address: str | None = None
    # the following attributes are automatically processed and set
    sequence: dict | None = None
    sync_en: bool | None = None
    offset_awg_path0: float | None = None
    offset_awg_path1: float | None = None
    marker_ovr_en: bool | None = None
    marker_ovr_value: int | None = None
    integration_length_acq: int | None = None
    thresholded_acq_rotation: float | None = None
    thresholded_acq_threshold: float | None = None
    demod_en_acq: bool | None = None
    nco_freq: int | None = None
    mod_en_awg: bool | None = None
    mixer_corr_gain_ratio: float | None = None
    mixer_corr_phase_offset_degree: float | None = None

    @classmethod
    def build(
        cls,
        address: PortAddress,
        channel_id: ChannelId,
        channels: dict[ChannelId, Channel],
        configs: Configs,
        acquisition: AcquisitionType,
        rf: bool,
        sequence: Q1Sequence | None = None,
    ) -> "SequencerConfig":
        config = configs[channel_id]

        # conditional configurations
        cfg = cls(
            # connect to physical address
            address=address.local_address,
            # TODO: mixer calibration not yet propagated
            offset_awg_path0=0.0,
            offset_awg_path1=0.0,
            # TODO: properly document - the first 4 marker bits are used to toggle
            # outputs, enabling suitable amplification
            marker_ovr_en=True,
            marker_ovr_value=15,
            # upload sequence
            # - ensure JSON compatibility of the sent dictionary
            sequence=(
                json.loads(sequence.model_dump_json()) if sequence is not None else None
            ),
            # configure the sequencers to synchronize
            sync_en=True,
            # modulation, only disable for QCM - always used for flux pulses
            mod_en_awg=rf,
        )

        # acquisition
        if address.input:
            assert isinstance(config, AcquisitionConfig)
            length = _integration_length(sequence) if sequence is not None else None
            if length is not None:
                cfg.integration_length_acq = length
            # discrimination
            if config.iq_angle is not None:
                cfg.thresholded_acq_rotation = np.degrees(config.iq_angle % (2 * np.pi))
            if config.threshold is not None and length is not None:
                # threshold needs to be compensated by length
                # see: https://docs.qblox.com/en/main/api_reference/sequencer.html#Sequencer.thresholded_acq_threshold
                cfg.thresholded_acq_threshold = config.threshold * length
            # demodulation
            cfg.demod_en_acq = acquisition is not AcquisitionType.RAW

        # set NCO frequency
        # note that probe channels also include readout ones (probe+acquisition), thus
        # there is no need to set it separately for the acquisition (which is on the
        # same IO sequencer)
        probe = channels[channel_id].iqout(channel_id)
        if probe is not None:
            freq = cast(IqConfig, configs[probe]).frequency
            probe_ = cast(IqChannel, channels[probe])
            assert probe_.lo is not None
            lo_freq = cast(OscillatorConfig, configs[probe_.lo]).frequency
            cfg.nco_freq = int(freq - lo_freq)
            assert probe_.mixer is not None
            mixer = cast(IqMixerConfig, configs[probe_.mixer])
            cfg.mixer_corr_gain_ratio = mixer.scale_q
            cfg.mixer_corr_phase_offset_degree = mixer.phase_q

        return cfg

    def apply(self, seq: Sequencer):
        """Configure sequencer-wide settings."""
        if self.address is not None:
            seq.connect_sequencer(self.address)

        # values already applied
        applied = {"address"}
        for name in self.model_fields_set - applied:
            value = getattr(self, name)
            if value is not None:
                seq.parameters[name].set(value)
