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
                sequencers = self._configure(
                    sequences_, configs, options_.acquisition_type
                )
                log.status(self.cluster, sequencers)

                # finally execute the experiment, and fetch results
                duration = options_.estimate_duration([ps], sweepers_)
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

    def _configure(
        self,
        sequences: dict[ChannelId, Q1Sequence],
        configs: Configs,
        acquisition: AcquisitionType,
    ) -> SequencerMap:
        """Configure modules and sequencers.

        The return value consists of the association map from channels
        to sequencers, for each module.
        """
        sequencers = defaultdict(dict)
        for slot, chs in self._channels_by_module.items():
            module = self._modules[slot]

            # each channel is going to be assigned to its own sequencer, thus they can
            # not be outnumbered
            assert len(module.sequencers) >= len(chs)

            ids = [ch for ch, _ in chs]
            channels = {id: ch for id, ch in self.channels.items() if id in ids}
            # compute module configurations, and apply them
            los = config.module.los(self._los, configs, ids)
            config.ModuleConfig.build(channels, configs, los, module.is_qrm_type).apply(
                module
            )

            # configure all sequencers, and store active ones' association to channels
            rf = module.is_rf_type
            for idx, ((ch, address), sequencer) in enumerate(
                zip(chs, module.sequencers)
            ):
                seq = sequences.get(ch, Q1Sequence.empty())
                config.SequencerConfig.build(
                    address, seq, ch, self.channels, configs, acquisition, idx, rf
                ).apply(sequencer)
                # only collect active sequencers
                if not seq.is_empty:
                    sequencers[slot][ch] = idx

        return sequencers

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

    @cached_property
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
                for ch, iq in ((ch, channels[ch].iqout(ch)) for ch in self.channels)
                if iq is not None
            )
            if lo is not None
        }
