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
from qibolab._core.instruments.qblox import validate
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
        # We can disconnect the module only when initializing it for the first time (for static configurations).
        modcfg.disconnect_module(module)
        modcfg.update_module(module)
    return modcfgs


def _configure_sequencers(
    cluster: Cluster,
    configs: Configs,
    acquisition: AcquisitionType = AcquisitionType.INTEGRATION,
    sequences: dict[ChannelId, Q1Sequence] | None = None,
) -> tuple[
    SequencerMap,
    dict[SlotId, dict[int, SequencerConfig]],
    dict[SlotId, list[int]],
]:
    """Configure sequencers.

    Returns the channel-to-sequencer association map, the sequencer
    configurations grouped by module and sequencer index, and the indices of
    RAW acquisition sequencers grouped by module.

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
            # populate channel-to-sequencer mapping
            sequencers[slot][ch] = idx

    return sequencers, seqcfgs, acquisition_sequencers


def _enable_raw_acquisition(
    cluster: Cluster, acq_sequencers: dict[SlotId, list[int]]
) -> None:
    """Configure scope acquisition settings on modules with RAW inputs.

    For each module with acquisition sequencers, enable scope acquisition
    averaging on both paths and select each listed sequencer for scope
    acquisition. RAW acquisition is restricted to cyclic averaging because of
    scope acquisition memory limits.
    """
    validate.validate_raw_acquisition_sequencers(acq_sequencers)

    for slot, seq_indices in acq_sequencers.items():
        readout_module = cluster._modules[slot]
        readout_module.scope_acq_sequencer_select(seq_indices[0])

        readout_module.parameters["scope_acq_avg_mode_en_path0"].set(True)
        readout_module.parameters["scope_acq_avg_mode_en_path1"].set(True)


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
    :meth:`_enable_raw_acquisition` to dynamically reconfigure the relevant
    readout modules so that each acquisition sequencer is individually
    selected for scope acquisition mode. For other acquisition types
    (e.g. ``INTEGRATION``), only the standard sequencer configuration is
    applied.

    Returns the channel-to-sequencer association map, grouped by module.
    """

    sequencers, _, acquisition_sequencers = _configure_sequencers(
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
