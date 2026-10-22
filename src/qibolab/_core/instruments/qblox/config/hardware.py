from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from qibolab._core.components import (
    Configs,
)
from qibolab._core.execution_parameters import (
    AcquisitionType,
)
from qibolab._core.identifier import ChannelId
from qibolab._core.instruments.qblox.identifiers import SequencerMap, SlotId
from qibolab._core.instruments.qblox.sequence import Q1Sequence

from . import module as mo
from .sequencer import SequencerConfig

if TYPE_CHECKING:
    from qibolab._core.instruments.qblox.cluster import Cluster


def configure_modules_static(
    cluster: Cluster, configs: Configs
) -> dict[SlotId, mo.ModuleConfig]:
    """Configure module settings (e.g. LOs, Mixers).

    Returns a dictionary mapping slots to their respective ModuleConfig.
    """
    modcfgs = {}
    for slot, chs in cluster._channels_by_module.items():
        module = cluster._modules[slot]
        ids = {id for id, _ in chs}
        channels = {id: ch for id, ch in cluster.channels.items() if id in ids}
        los = mo.los(cluster._los, configs, ids)
        mixers = mo.mixers(cluster._mixers, configs, ids)
        modcfg = modcfgs[slot] = mo.ModuleConfig.build(
            channels,
            configs,
            los,
            mixers,
            is_qcm_non_rf_type=module.is_qcm_type and not module.is_rf_type,
        )
        modcfg.apply(module)
    return modcfgs


def _configure_sequencers(
    cluster: Cluster,
    configs: Configs,
    acquisition: AcquisitionType = AcquisitionType.INTEGRATION,
    sequences: dict[ChannelId, Q1Sequence] | None = None,
) -> tuple[
    SequencerMap, dict[SlotId, dict[int, SequencerConfig]], dict[SlotId, list[int]]
]:
    """Configure sequencers.

    The return value consists of the association map from channels to sequencers,
    for each module. And the sequencer configurations.

    For configuration testing purpose, it is possible to also configure sequencers
    with no sequence provided. In which case, it will attempt to assign sequencers
    to all available channels (as opposed to just those involved in the experiment,
    and thus in the sequences). For the sake of simplifying the usage of this
    function, a default acquisition type is provided
    (:attr:`AcquisitionType.INTEGRATION`). The only true alternative to this value
    is :attr:`AcquisitionType.RAW`, since further configurations are required to
    operate in scope mode.
    """
    sequencers = defaultdict(dict)
    acquisition_sequencers = defaultdict(list)
    exec_mode = sequences is not None
    sequences_ = defaultdict(lambda: None, sequences if exec_mode else {})

    seqcfgs = {}
    for slot, chs in cluster._channels_by_module.items():
        module = cluster._modules[slot]

        # each channel is going to be assigned to its own sequencer, thus they can
        # not be outnumbered
        assert len(module.sequencers) >= len(chs)

        # configure all sequencers, and store association to channels
        seqcfgs[slot] = {}
        for idx, ((ch, address), sequencer) in enumerate(zip(chs, module.sequencers)):
            # Only configure and register sequencer for active channels. For passive
            # channels the sequencer operations are not relevant, e.g. a flux
            # channel with no registered pulse will still set an offset, but this
            # will happen at port level, and it is consumed in the `ModuleConfig` if
            # not in execution mode, configure all channels, to test the
            # configuration itself.
            if exec_mode and ch not in sequences:
                continue

            seqcfg = seqcfgs[slot][idx] = SequencerConfig.build(
                address,
                ch,
                cluster.channels,
                configs,
                acquisition,
                module.is_rf_type,
                sequence=sequences_[ch],
            )
            seqcfg.apply(sequencer)

            # Only RAW input sequencers set this flag to False; output
            # sequencers leave it as None and cannot trigger scope capture.
            if seqcfg.demod_en_acq is False:
                acquisition_sequencers[slot].append(idx)
                if len(acquisition_sequencers[slot]) > 1:
                    raise ValueError(
                        "RAW acquisition supports only one acquisition sequencer per "
                        f"module, but module in slot {slot} has {len(acquisition_sequencers[slot])}: "
                        f"{acquisition_sequencers[slot]}."
                    )

            # populate channel-to-sequencer mapping
            sequencers[slot][ch] = idx

    return sequencers, seqcfgs, acquisition_sequencers


def _enable_raw_acquisition(
    cluster: Cluster, acq_sequencers: dict[SlotId, list[int]]
) -> None:
    """Select the scope acquisition sequencer on modules with RAW inputs.

    Scope acquisition is module-wide, thus at most one sequencer per module can be
    selected.
    Scope averaging is instead enabled on all QRM modules by the
    :class:`ModuleConfig` defaults, such that RAW traces are averaged over shots in
    hardware. Since the scope memory can not retain a separate trace per shot, RAW
    acquisition is restricted to cyclic averaging (cf.
    :func:`validate.validate_raw_averaging_mode`).
    """

    for slot, seq_indices in acq_sequencers.items():
        cluster._modules[slot].scope_acq_sequencer_select(seq_indices[0])


def configure_hardware_per_sequence(
    cluster: Cluster,
    configs: Configs,
    acquisition: AcquisitionType,
    sequences: dict[ChannelId, Q1Sequence] | None = None,
) -> SequencerMap:
    """Configure the hardware for a specific sequence.

    This method configures sequencers for the provided sequences (or all
    available channels when ``sequences`` is ``None``) and applies any
    acquisition-specific module settings.

    For ``AcquisitionType.RAW`` acquisitions, the method additionally calls
    :func:`_enable_raw_acquisition` to select the acquisition sequencer of each
    readout module for scope acquisition. For other acquisition types
    (e.g. ``INTEGRATION``), only the standard sequencer configuration is applied.

    Returns the channel-to-sequencer association map, grouped by module.
    """
    sequencers, _seqcfgs, acquisition_sequencers = _configure_sequencers(
        cluster=cluster,
        configs=configs,
        acquisition=acquisition,
        sequences=sequences,
    )

    # at the time being we only dynamically change the modules for RAW acquisition
    # but in the future it can be expanded
    if acquisition is AcquisitionType.RAW:
        _enable_raw_acquisition(cluster=cluster, acq_sequencers=acquisition_sequencers)

    return sequencers
