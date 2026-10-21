import logging
import time
import warnings
from collections import defaultdict
from functools import cached_property
from itertools import groupby
from typing import cast

import qblox_instruments as qblox
from qblox_instruments.qcodes_drivers.module import Module
from qblox_instruments.qcodes_drivers.sequencer import Sequencer
from qcodes.instrument import find_or_create_instrument

from qibolab._core.components import (
    AcquisitionChannel,
    Configs,
    DcChannel,
    DcConfig,
    IqChannel,
)
from qibolab._core.execution_parameters import (
    AcquisitionType,
    ExecutionParameters,
)
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses.pulse import PulseId, Readout
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import Model
from qibolab._core.sweeper import ParallelSweepers, Parameter, normalize_sweepers

from . import config
from .batching import batch_sequences_by_cluster_memory_limits
from .config import PortAddress
from .identifiers import SequencerMap, SlotId
from .log import Logger
from .results import AcquiredData, extract, integration_lengths
from .sequence import Q1Sequence, compile
from .utils import (
    batch_shots,
    concat_shots,
    lo_configs,
    time_of_flights,
)
from .validate import (
    assert_channels_exclusion,
    validate_sequence,
)

logger = logging.getLogger(__name__)


__all__ = ["Cluster"]

SAMPLING_RATE = 1


def _compute_duration(
    ps: PulseSequence,
    sweepers: list[ParallelSweepers],
    options: ExecutionParameters,
    configs: Configs,
) -> float:
    """Compute the total program duration including time of flight and
    synchronization waiting time.
    """

    # TODO: include the time of flight calculation at the level of
    # Platform.execute rather than in the qblox driver. This will require
    # propagating the changes also to qibocal.
    time_of_flight = max(
        [time_of_flights(configs)[ch[0]] for ch in ps if hasattr(ch[1], "acquisition")],
        default=0.0,
    )

    # TODO: wait_sync duration is determined as explained in this comment
    # https://github.com/qiboteam/qibolab/pull/1389#issuecomment-3884129213.
    # It should be checked with Qblox if the sync time can indeed be of the
    # order of 900 ns.
    wait_sync_duration = 900
    duration = options.estimate_duration(
        [ps], sweepers, time_of_flight + wait_sync_duration
    )
    return duration


def _add_time_of_flight(sequence: PulseSequence, configs: Configs) -> PulseSequence:
    time_of_flights_ = time_of_flights(configs)
    return PulseSequence(
        [
            (
                ch,
                ev
                if not isinstance(ev, Readout)
                else ev.model_copy(update={"time_of_flight": time_of_flights_[ch]}),
            )
            for ch, ev in sequence
        ]
    )


def _batch_sequences(
    sequences: list[PulseSequence],
    sweepers: list[ParallelSweepers],
    options: ExecutionParameters,
    qcm_channels: set[ChannelId],
    qrm_channels: set[ChannelId],
    configs: Configs,
) -> list[PulseSequence]:
    batched_seqs = (
        batch_sequences_by_cluster_memory_limits(
            sequences,
            sweepers,
            options,
            qcm_channels,
            qrm_channels,
        )
        if options.averaging_mode.average
        else sequences
    )
    return [_add_time_of_flight(b, configs).align_to_delays() for b in batched_seqs]


def _merge_phases_if_no_phase_sweeper(
    sweepers: list[ParallelSweepers],
    sequences: list[PulseSequence],
) -> tuple[list[PulseSequence], bool]:
    """
    Process pulse sequences based on the presence of phase sweepers.

    If any sweeper in the provided list is a phase or relative_phase sweeper,
    the sequences are returned unchanged. Otherwise, the phases in the sequences
    are summed to simplify the pulse sequences.
    """
    phase_sweeper_present = any(
        sweeper.parameter in {Parameter.relative_phase, Parameter.phase}
        for parallel_sweepers in sweepers
        for sweeper in parallel_sweepers
    )
    return [
        ps if phase_sweeper_present else ps.to_vzs().collect_vzs() for ps in sequences
    ], phase_sweeper_present


class ClusterConfigs(Model):
    modules: dict[int, config.ModuleConfig]
    sequencers: dict[int, dict[int, config.SequencerConfig]]


class Cluster(Controller):
    """Controller object for Qblox cluster."""

    name: str
    """Device name.

    As described in:
    https://docs.qblox.com/en/main/getting_started/setup.html#connecting-to-multiple-instruments
    """
    _cluster: qblox.Cluster | None = None
    # Cache of the last applied sequencer configuration per (slot, sequencer index), and
    # the active sequencer indices per slot. Invalidated on connect/reset/disconnect.
    _applied_seqcfg: dict | None = None
    _active_seqs: dict | None = None

    @property
    def cluster(self) -> qblox.Cluster:
        """Ensure cluster object access.

        The object presence results from an existing connection.
        """
        assert self._cluster is not None
        return self._cluster

    @property
    def sampling_rate(self) -> int:
        """Provide instrument's sampling rate."""
        return SAMPLING_RATE

    @property
    def is_connected(self) -> bool:
        """Determine connections status."""
        return self._cluster is not None

    def reset(self) -> None:
        """Reset cluster parameters."""
        assert self._cluster is not None
        # The hardware reset clears all sequencer connections and settings, so
        # the cached applied state is no longer valid.
        self._applied_seqcfg = {}
        self._active_seqs = {}
        self._cluster.reset()

    def connect(self) -> None:
        """Connect and initialize the instrument."""
        if self.is_connected:
            return

        self._cluster = find_or_create_instrument(
            qblox.Cluster, recreate=True, name=self.name, identifier=self.address
        )
        self._invalidate_modules()
        self._applied_seqcfg = {}
        self._active_seqs = {}

    def disconnect(self) -> None:
        """Disconnect and reset the instrument."""
        assert self._cluster is not None

        # Ensure static flux offsets are reset before closing the connection to avoid
        # heating the fridge.
        self._set_dc_offsets_to_0()

        for module in self._modules.values():
            module.stop_sequencer()
        self._cluster.close()
        self._cluster = None
        self._invalidate_modules()
        self._applied_seqcfg = None
        self._active_seqs = None

    def _set_dc_offsets_to_0(self) -> None:
        flux_slot_to_ports: dict[SlotId, set[int]] = defaultdict(set)
        for channel in self.channels.values():
            if not isinstance(channel, DcChannel):
                continue
            address = PortAddress.from_path(channel.path)
            flux_slot_to_ports[address.slot].add(address.ports[0] - 1)

        for slot, ports in flux_slot_to_ports.items():
            module = self._modules[slot]
            for port in ports:
                offset_parameter = f"out{port}_offset"
                module.parameters[offset_parameter].set(0.0)

    def play(
        self,
        configs: Configs,
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[PulseId, Result]:
        """Execute the given experiment."""

        self.reset()

        processed_sequences, phase_sweeper_present = _merge_phases_if_no_phase_sweeper(
            sweepers, sequences
        )

        # If acquisition is cyclic (averaging over shots on hardware), we combine as
        # many sequences as possible in a single batch, according to the cluster
        # memory limits.
        qrm_slots = {
            slot for slot, module in self._modules.items() if module.is_qrm_type
        }
        qrm_channels = {
            channelid
            for channelid, channelobj in self.channels.items()
            if PortAddress.from_path(channelobj.path).slot in qrm_slots
        }
        qcm_channels = set(self.channels) - qrm_channels
        sweepers_ = normalize_sweepers(
            sweepers,
            lo_configs(self._los, configs),
            {
                ch: cfg.offset
                for ch, cfg in configs.items()
                if isinstance(cfg, DcConfig)
            },
        )

        batched_seqs = _batch_sequences(
            processed_sequences, sweepers_, options, qcm_channels, qrm_channels, configs
        )

        # configure the modules
        _module_configs = self._configure_modules(configs)

        # Execute each batch sequentially, and concatenate results
        log = Logger(configs)
        results = {}
        for ps in batched_seqs:
            psres = []
            for shots in batch_shots(ps, sweepers_, options):
                options_ = options.model_copy(update={"nshots": shots})

                # first compile pulses and sweepers into Qblox sequences
                assert_channels_exclusion(ps, self._probes)
                sequences_ = compile(
                    ps,
                    sweepers_,
                    options_,
                    self.sampling_rate,
                    merged_vzs=not phase_sweeper_present,
                )

                for channelid, seq in sequences_.items():
                    slot = PortAddress.from_path(self.channels[channelid].path).slot
                    validate_sequence(seq, self._modules[slot].is_qrm_type)

                log.sequences(sequences_)

                # then configure sequencers (including sequences upload).
                sequencers, _sequencer_configs = self._configure_sequencers(
                    configs=configs,
                    acquisition=options_.acquisition_type,
                    sequences=sequences_,
                )
                log.status(self.cluster, sequencers)

                # finally execute the experiment, and fetch results
                duration = _compute_duration(ps, sweepers_, options_, configs)
                logger.info(f"Qblox expected execution time: {duration:.3f} s")
                data = self._execute(
                    sequencers, sequences_, duration, options_.acquisition_type
                )
                log.data(data)

                # Process raw results to adhere to the standard format. Only needed if
                # AcquisitionType.INTEGRATION.
                lengths = (
                    integration_lengths(sequences_, sequencers, self._modules)
                    if options_.acquisition_type is AcquisitionType.INTEGRATION
                    else {}
                )
                psres.append(
                    extract(
                        data,
                        lengths,
                        options_.acquisition_type,
                        options_.results_shape(sweepers_),
                    )
                )

            # update results - concatenating shots, if needed
            results |= concat_shots(psres, options)
        return results

    def _disconnect_slots(self, slots: set[SlotId]) -> None:
        """Disconnect outputs/inputs and desync the sequencers of the given slots.

        see FAQ in docs for disconnect_outputs() and disconnect_inputs()
        https://docs.qblox.com/en/v2025.12.0/products/qblox_instruments/faq.html
        """
        for slot in slots:
            module = self._modules[slot]
            # Disconnect outputs because `SequencerConfig.apply` creates new connections
            # between the indexed sequencer and various inputs/outputs. Existing
            # connections would otherwise cause the operation to fail.
            module.disconnect_outputs()
            if module.is_qrm_type:
                module.disconnect_inputs()

            for seq in module.sequencers:
                # Disable synchronization. This is needed because different
                # sequences may use different sequencers, and if a previous sequence
                # used a sequencer that is not used in the current one, it can cause
                # an indefinite wait during `wait_sync`.
                seq.sync_en(False)

    def _configure_modules(self, configs: Configs) -> dict[SlotId, config.ModuleConfig]:
        """Configure module settings (e.g. LOs, Mixers).

        Returns a dictionary mapping slots to their respective ModuleConfig.
        """
        modcfgs = {}
        for slot, chs in self._channels_by_module.items():
            module = self._modules[slot]
            ids = {id for id, _ in chs}
            channels = {id: ch for id, ch in self.channels.items() if id in ids}
            los = config.module.los(self._los, configs, ids)
            mixers = config.module.mixers(self._mixers, configs, ids)
            modcfg = modcfgs[slot] = config.ModuleConfig.build(
                channels,
                configs,
                los,
                mixers,
                is_qcm_non_rf_type=module.is_qcm_type and not module.is_rf_type,
            )
            modcfg.apply(module)
        return modcfgs

    def _build_sequencer_configs(
        self,
        configs: Configs,
        acquisition: AcquisitionType,
        sequences: dict[ChannelId, Q1Sequence] | None,
    ) -> tuple[
        dict[SlotId, dict[int, config.SequencerConfig]],
        dict[tuple[SlotId, int], tuple[config.SequencerConfig, Sequencer, ChannelId]],
        dict[SlotId, set[int]],
    ]:
        """Build sequencer configs for all active channels.

        Returns the per-slot config map, a flat map of (slot, idx) to (config,
        sequencer, channel) for active sequencers, and the active sequencer indices per
        slot.
        """
        exec_mode = sequences is not None
        sequences_ = defaultdict(lambda: None, sequences if exec_mode else {})

        seqcfgs: dict[SlotId, dict[int, config.SequencerConfig]] = {}
        new_cfgs: dict[
            tuple[SlotId, int],
            tuple[config.SequencerConfig, Sequencer, ChannelId],
        ] = {}
        cur_active: dict[SlotId, set[int]] = defaultdict(set)
        for slot, chs in self._channels_by_module.items():
            module = self._modules[slot]
            assert len(module.sequencers) >= len(chs)

            seqcfgs[slot] = {}
            for idx, ((ch, address), sequencer) in enumerate(
                zip(chs, module.sequencers)
            ):
                # In execution mode, skip passive channels (no pulse); their offsets are
                # handled at port level via `ModuleConfig`. Without sequences, configure
                # all channels to test the config.
                if exec_mode and ch not in sequences:
                    continue

                seqcfg = seqcfgs[slot][idx] = config.SequencerConfig.build(
                    address,
                    ch,
                    self.channels,
                    configs,
                    acquisition,
                    module.is_rf_type,
                    sequence=sequences_[ch],
                )
                new_cfgs[(slot, idx)] = (seqcfg, sequencer, ch)
                cur_active[slot].add(idx)

        return seqcfgs, new_cfgs, cur_active

    def _slots_to_disconnect(
        self,
        new_cfgs: dict[
            tuple[SlotId, int],
            tuple[config.SequencerConfig, Sequencer, ChannelId],
        ],
        cur_active: dict[SlotId, set[int]],
    ) -> set[SlotId]:
        """Determine which slots need a full disconnect+desync before (re)applying.

        A slot needs it when:
        - its active sequencer set changed (different channels mapped to
          sequencers, so old connections/sync state are stale),
        - a sequencer's connection address changed,
        - a previously set parameter is now None (apply_changed can't clear it,
          so the stale value would persist on the hardware), or
        - the slot was active before but is not now (stale connections/sync).
        """
        applied = self._applied_seqcfg
        prev_active = self._active_seqs
        slots_to_disconnect: set[SlotId] = set()
        for slot, idxs in cur_active.items():
            prev_idxs = (
                prev_active.get(slot, set()) if prev_active is not None else set()
            )
            if idxs != prev_idxs:
                slots_to_disconnect.add(slot)
                continue
            for idx in idxs:
                old = applied.get((slot, idx)) if applied is not None else None
                new = new_cfgs[(slot, idx)][0]
                needs_disconnect = (
                    old is None
                    or old.address != new.address
                    or any(
                        getattr(old, name) is not None and getattr(new, name) is None
                        for name in type(new).model_fields
                        if name != "address"
                    )
                )
                if needs_disconnect:
                    slots_to_disconnect.add(slot)
                    break
        if prev_active is not None:
            for slot in prev_active:
                if slot not in cur_active:
                    slots_to_disconnect.add(slot)
        return slots_to_disconnect

    def _configure_sequencers(
        self,
        configs: Configs,
        acquisition: AcquisitionType = AcquisitionType.INTEGRATION,
        sequences: dict[ChannelId, Q1Sequence] | None = None,
    ) -> tuple[SequencerMap, dict[SlotId, dict[int, config.SequencerConfig]]]:
        """Configure sequencers.

        Returns the channel-to-sequencer association map per module and the
        sequencer configurations.

        Without ``sequences``, all channels are configured (useful for testing).
        When the channel map is unchanged since the previous call, existing
        connections are reused and only changed parameters are re-applied.
        """
        sequencers = defaultdict(dict)
        seqcfgs, new_cfgs, cur_active = self._build_sequencer_configs(
            configs, acquisition, sequences
        )

        # Disconnect is module-level, so after it all active sequencers in the slot must
        # be fully re-applied.
        slots_to_disconnect = self._slots_to_disconnect(new_cfgs, cur_active)
        self._disconnect_slots(slots_to_disconnect)

        # Apply the configurations: a full (re)connect when the slot was disconnected,
        # the address changed, or on first use; otherwise only the changed parameters.
        applied = self._applied_seqcfg
        for (slot, idx), (seqcfg, sequencer, ch) in new_cfgs.items():
            old = applied.get((slot, idx)) if applied is not None else None
            if (
                slot in slots_to_disconnect
                or old is None
                or old.address != seqcfg.address
            ):
                seqcfg.apply(sequencer)
            else:
                seqcfg.apply_changed(sequencer, old)
            sequencers[slot][ch] = idx
            if self._applied_seqcfg is not None:
                self._applied_seqcfg[(slot, idx)] = seqcfg

        if self._active_seqs is not None:
            self._active_seqs = dict(cur_active)

        return sequencers, seqcfgs

    def configure(
        self,
        configs: Configs,
        acquisition: AcquisitionType = AcquisitionType.INTEGRATION,
        sequences: dict[ChannelId, Q1Sequence] | None = None,
    ) -> tuple[SequencerMap, ClusterConfigs]:
        """
        .. deprecated:: 0.2.16
            Use `configure_sequencers` and/or `_configure_modules` instead.
        """
        modules_configs = self._configure_modules(configs)
        sequencers, seqcfgs = self._configure_sequencers(
            configs, acquisition, sequences
        )
        return sequencers, ClusterConfigs(modules=modules_configs, sequencers=seqcfgs)

    def _execute(
        self,
        sequencers: SequencerMap,
        sequences: dict[ChannelId, Q1Sequence],
        duration: float,
        acquisition: AcquisitionType,
    ) -> dict[ChannelId, AcquiredData]:
        """Execute experiment and fetch results."""
        # start experiment, arming and starting all sequencers
        for mod, seqs in sequencers.items():
            module = self._modules[mod]
            for seq in seqs.values():
                module.arm_sequencer(seq)
            module.start_sequencer()

        # wait for experiment completion
        time.sleep(duration)

        # fetch acquired results
        acquisitions = {}
        for slot, seqs in sequencers.items():
            for ch, seq in seqs.items():
                # wait all sequencers
                status = self.cluster.get_sequencer_status(slot, seq, timeout=1)
                if status.status is qblox.SequencerStatuses.ERROR:
                    raise RuntimeError(f"slot: {slot}, seq: {seq}\n{status}")
                if status.status is qblox.SequencerStatuses.WARNING:
                    warnings.warn(f"slot: {slot}, seq: {seq}\n{status}")

                # skip results retrieval for passive or inactive sequencers...
                sequence = sequences.get(ch)
                if sequence is None:
                    continue
                # ... and also for channels missing acquisition instructions
                seq_acqs = sequence.acquisitions
                if len(seq_acqs) == 0:
                    continue

                # ensure acquisition status, and fetch results
                acq_status = self.cluster.get_acquisition_status(slot, seq, timeout=1)
                if acq_status is False:
                    raise RuntimeError(
                        f"Acquisition not completed. slot: {slot}, seq: {seq}."
                    )
                if acquisition is AcquisitionType.RAW:
                    for name in seq_acqs:
                        self.cluster.store_scope_acquisition(slot, seq, str(name))
                acquisitions[ch] = self.cluster.get_acquisitions(slot, seq)

        return acquisitions

    @cached_property
    def _modules(self) -> dict[SlotId, Module]:
        """Retrieve slot to module object mapping."""
        return {mod.slot_idx: mod for mod in self.cluster.modules if mod.present()}

    def _invalidate_modules(self) -> None:
        """Drop the cached `_modules`."""
        if "_modules" in self.__dict__:
            delattr(self, "_modules")

    @cached_property
    def _probes(self) -> set[ChannelId]:
        """Determine probe channels."""
        return {
            ch.probe
            for ch in self.channels.values()
            if isinstance(ch, AcquisitionChannel) and ch.probe is not None
        }

    @cached_property
    def _channels_by_module(self) -> dict[SlotId, list[tuple[ChannelId, PortAddress]]]:
        """Identify channels associated to each module.

        Channels are otherwise a set, where the association is stored in
        the :attr:`qibolab.Channel.port` attributes of each channel.
        """
        addresses = {
            name: PortAddress.from_path(ch.path) for name, ch in self.channels.items()
        }
        return {
            k: [el[1] for el in g]
            for k, g in groupby(
                sorted(
                    (address.slot, (ch, address))
                    for ch, address in addresses.items()
                    if ch not in self._probes
                ),
                key=lambda el: el[0],
            )
        }

    @cached_property
    def _los(self) -> dict[ChannelId, str]:
        """Extract channel to LO mapping.

        The result contains the associated channel, since required to
        address the LO through the API. While the LO identifier is used
        to retrieve the configuration.
        """
        channels = self.channels
        return {
            ch: lo
            for ch, lo in (
                (ch, cast(IqChannel, channels[iq]).lo)
                for ch, iq in ((ch, channels[ch].iqout(ch)) for ch in channels)
                if iq is not None
            )
            if lo is not None
        }

    @cached_property
    def _mixers(self) -> dict[ChannelId, str]:
        """Extract channel to mixer mapping."""
        # TODO: identical to the `._los` property, deduplicate it please...
        channels = self.channels
        return {
            ch: mix
            for ch, mix in (
                (ch, cast(IqChannel, channels[iq]).mixer)
                for ch, iq in ((ch, channels[ch].iqout(ch)) for ch in channels)
                if iq is not None
            )
            if mix is not None
        }
