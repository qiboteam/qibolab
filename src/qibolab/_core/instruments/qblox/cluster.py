import time
import warnings
from collections import defaultdict
from functools import cached_property
from itertools import groupby
from typing import Optional, cast

import qblox_instruments as qblox
from qblox_instruments.qcodes_drivers.module import Module
from qcodes.instrument import find_or_create_instrument

from qibolab._core.components import AcquisitionChannel, Configs, DcConfig, IqChannel
from qibolab._core.execution_parameters import AcquisitionType, ExecutionParameters
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses.pulse import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import Model
from qibolab._core.sweeper import ParallelSweepers, normalize_sweepers

from . import config
from .config import PortAddress
from .identifiers import SequencerMap, SlotId
from .log import Logger
from .results import AcquiredData, extract, integration_lenghts
from .sequence import Q1Sequence, compile
from .utils import batch_shots, concat_shots, lo_configs, time_of_flights
from .validate import assert_channels_exclusion, validate_sequence

__all__ = ["Cluster"]

SAMPLING_RATE = 1


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
    bounds: str = "qblox/bounds"
    _cluster: Optional[qblox.Cluster] = None

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

    def connect(self):
        """Connect and initialize the instrument."""
        if self.is_connected:
            return

        self._cluster = find_or_create_instrument(
            qblox.Cluster, recreate=True, name=self.name, identifier=self.address
        )

    def disconnect(self):
        """Disconnect and reset the instrument."""
        assert self._cluster is not None

        for module in self._modules.values():
            module.stop_sequencer()
        self._cluster.reset()
        self._cluster.close()
        self._cluster = None

    def play(
        self,
        configs: Configs,
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[PulseId, Result]:
        """Execute the given experiment."""
        results = {}
        log = Logger(configs)

        # no unrolling yet: act one sequence at a time, and merge results
        for ps in sequences:
            # full reset of the cluster, to erase leftover configurations and sequencer
            # synchronization registration
            # NOTE: until not unrolled, each sequence execution should be independent
            # TODO: once unrolled, this reset should be preserved, since it is required
            # for multiple experiments sharing the same connection
            self.reset()
            # split shots in batches, in case the required experiment exceeds the
            # allowed memory
            psres = []
            for shots in batch_shots(ps, sweepers, options):
                options_ = options.model_copy(update={"nshots": shots})
                sweepers_ = normalize_sweepers(
                    sweepers,
                    lo_configs(self._los, configs),
                    {
                        ch: cfg.offset
                        for ch, cfg in configs.items()
                        if isinstance(cfg, DcConfig)
                    },
                )
                # first compile pulses and sweepers into Qblox sequences
                assert_channels_exclusion(ps, self._probes)
                sequences_ = compile(
                    ps,
                    sweepers_,
                    options_,
                    self.sampling_rate,
                    time_of_flights(configs),
                )
                for seq in sequences_.values():
                    validate_sequence(seq)
                log.sequences(sequences_)

                # then configure modules and sequencers
                # (including sequences upload)
                sequencers, _ = self.configure(
                    configs, options_.acquisition_type, sequences=sequences_
                )
                log.status(self.cluster, sequencers)

                # TODO: include the time of flight calculation at the level of
                # Platform.execute rather than in the qblox driver. This will require
                # propagating the changes also to qibocal.
                time_of_flight = max(
                    [
                        time_of_flights(configs)[ch[0]]
                        for ch in ps
                        if hasattr(ch[1], "acquisition")
                    ],
                    default=0.0,
                )
                duration = options_.estimate_duration([ps], sweepers_, time_of_flight)
                # finally execute the experiment, and fetch results
                data = self._execute(
                    sequencers, sequences_, duration, options_.acquisition_type
                )
                log.data(data)

                # process raw results to adhere to standard format
                lenghts = integration_lenghts(sequences_, sequencers, self._modules)
                psres.append(
                    extract(
                        data,
                        lenghts,
                        options_.acquisition_type,
                        options_.results_shape(sweepers_),
                    )
                )

            # update results - concatenating shots, if needed
            results |= concat_shots(psres, options)
        return results

    def configure(
        self,
        configs: Configs,
        acquisition: AcquisitionType = AcquisitionType.INTEGRATION,
        sequences: Optional[dict[ChannelId, Q1Sequence]] = None,
    ) -> tuple[SequencerMap, ClusterConfigs]:
        """Configure modules and sequencers.

        The return value consists of the association map from channels
        to sequencers, for each module.

        For configuration testing purpose, it is possible to also configure modules and
        sequencers with no sequence provided. In which case, it will attempt to assign
        sequencers to all available channels (as opposed to just those involved in the
        experiment, and thus in the sequences).
        For the sake of simplifiying the usage of this function, a default acquisition
        type is provided (:attr:`AcquisitionType.INTEGRATION`). The only true
        alternative to this value is :attr:`AcquisitionType.RAW`, since further
        configurations are required to operate in scope mode.
        """
        sequencers = defaultdict(dict)
        exec_mode = sequences is not None
        sequences_ = defaultdict(lambda: None, sequences if exec_mode else {})

        modcfgs = {}
        seqcfgs = {}

        for slot, chs in self._channels_by_module.items():
            module = self._modules[slot]

            # each channel is going to be assigned to its own sequencer, thus they can
            # not be outnumbered
            assert len(module.sequencers) >= len(chs)

            ids = {id for id, _ in chs}
            channels = {id: ch for id, ch in self.channels.items() if id in ids}
            los = config.module.los(self._los, configs, ids)
            mixers = config.module.mixers(self._mixers, configs, ids)
            # compute module configurations, and apply them
            modcfg = modcfgs[slot] = config.ModuleConfig.build(
                channels, configs, los, mixers
            )
            modcfg.apply(module)
            seqcfgs[slot] = {}

            # configure all sequencers, and store association to channels
            rf = module.is_rf_type
            for idx, ((ch, address), sequencer) in enumerate(
                zip(chs, module.sequencers)
            ):
                # only configure and register sequencer for active channels
                # for passive channels the sequencer operations are not relevant, e.g. a
                # flux channel with no registered pulse will still set an offset, but
                # this will happen at port level, and it is consumed in the
                # `ModuleConfig` above
                # if not in execution mode, cnfigure all channels, to test the
                # configuration itself
                if exec_mode and ch not in sequences:
                    continue

                seqcfg = seqcfgs[slot][idx] = config.SequencerConfig.build(
                    address,
                    ch,
                    self.channels,
                    configs,
                    acquisition,
                    idx,
                    rf,
                    sequence=sequences_[ch],
                )
                seqcfg.apply(sequencer)
                # populate channel-to-sequencer mapping
                sequencers[slot][ch] = idx

        return sequencers, ClusterConfigs(modules=modcfgs, sequencers=seqcfgs)

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

        # approximately wait for experiment completion
        time.sleep(duration + 1)

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
                self.cluster.get_acquisition_status(slot, seq, timeout=1)
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
