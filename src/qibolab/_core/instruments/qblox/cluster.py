import logging
import time
import warnings
from collections import defaultdict
from functools import cached_property
from itertools import groupby
from typing import cast

import qblox_instruments as qblox
import scipy.constants as consts
from qblox_instruments.qcodes_drivers.module import Module
from qcodes.instrument import find_or_create_instrument

from qibolab._core.components import (
    AcquisitionChannel,
    Configs,
    DcChannel,
    DcConfig,
    IqChannel,
    IqConfig,
    OscillatorConfig,
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
    validate_raw_averaging_mode,
    validate_sequence,
)

logger = logging.getLogger(__name__)


__all__ = ["Cluster"]

SAMPLING_RATE = 1  # GS/s


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
        and options.acquisition_type is not AcquisitionType.RAW
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
        self._cluster.reset()

    def connect(self) -> None:
        """Connect and initialize the instrument."""
        if self.is_connected:
            return

        self._cluster = find_or_create_instrument(
            qblox.Cluster, recreate=True, name=self.name, identifier=self.address
        )

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

        validate_raw_averaging_mode(options)
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
        _module_configs = config.configure_modules_static(cluster=self, configs=configs)

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
                    validate_sequence(
                        sequence=seq,
                        is_qrm=self._modules[slot].is_qrm_type,
                        acquisition_type=options_.acquisition_type,
                    )

                log.sequences(sequences_)

                # then configure sequencers (including sequences upload)
                self._disconnect_and_desync_sequencers()
                # configure the hardware for the current sequence batch
                sequencers = config.configure_hardware_per_sequence(
                    cluster=self,
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

                # process raw results to adhere to standard format
                lengths = integration_lengths(sequences_, sequencers, self._modules)

                psres.append(
                    extract(
                        acquired_data=data,
                        lengths=lengths,
                        acquisition=options_.acquisition_type,
                        shape=options_.results_shape(sweepers_),
                        sampling_rate=self.sampling_rate * consts.giga,
                        frequencies=self._ifs(configs)
                        if options_.acquisition_type is AcquisitionType.RAW
                        else {},
                    )
                )

            # update results - concatenating shots, if needed
            results |= concat_shots(psres, options)
        return results

    def _disconnect_and_desync_sequencers(self) -> None:
        """Clear the relevant module settings before applying a new sequence."""
        # see FAQ in docs for disconnect_outputs() and disconnect_inputs()
        # https://docs.qblox.com/en/v2025.12.0/products/qblox_instruments/faq.html
        for module in self._modules.values():
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

    @property
    def _modules(self) -> dict[SlotId, Module]:
        """Retrieve slot to module object mapping."""
        return {mod.slot_idx: mod for mod in self.cluster.modules if mod.present()}

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

    def _ifs(self, configs: Configs) -> dict[ChannelId, float]:
        """Compute the intermediate frequencies for demodulation.

        IFs are computed in Hz.
        """
        frequencies = {}

        channels = self.channels
        los = self._los
        for ch, channel in channels.items():
            if not isinstance(channel, AcquisitionChannel) or channel.probe is None:
                continue

            probe = channel.probe
            lo = los.get(ch)
            lo_freq = cast(OscillatorConfig, configs[lo]).frequency if lo else 0.0
            frequencies[ch] = cast(IqConfig, configs[probe]).frequency - lo_freq
        return frequencies
