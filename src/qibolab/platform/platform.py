"""A platform for executing quantum algorithms."""

import dataclasses
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from math import prod
from typing import Any, Optional, TypeVar

import numpy as np
from qibo.config import log, raise_error

from qibolab.components import Config
from qibolab.execution_parameters import ConfigUpdate, ExecutionParameters
from qibolab.instruments.abstract import Controller, Instrument, InstrumentId
from qibolab.pulses import Delay, PulseSequence
from qibolab.qubits import Qubit, QubitId, QubitPair, QubitPairId
from qibolab.serialize_ import replace
from qibolab.sweeper import ParallelSweepers
from qibolab.unrolling import batch

InstrumentMap = dict[InstrumentId, Instrument]
QubitMap = dict[QubitId, Qubit]
CouplerMap = dict[QubitId, Qubit]
QubitPairMap = dict[QubitPairId, QubitPair]

IntegrationSetup = dict[str, tuple[np.ndarray, float]]

NS_TO_SEC = 1e-9

# TODO: replace with https://docs.python.org/3/reference/compound_stmts.html#type-params
T = TypeVar("T")


# TODO: lift for general usage in Qibolab
def default(value: Optional[T], default: T) -> T:
    """None replacement shortcut."""
    return value if value is not None else default


def unroll_sequences(
    sequences: list[PulseSequence], relaxation_time: int
) -> tuple[PulseSequence, dict[str, list[str]]]:
    """Unrolls a list of pulse sequences to a single sequence.

    The resulting sequence may contain multiple measurements.

    `relaxation_time` is the time in ns to wait for the qubit to relax between playing
    different sequences.

    It returns both the unrolled pulse sequence, and the map from original readout pulse
    serials to the unrolled readout pulse serials. Required to construct the results
    dictionary that is returned after execution.
    """
    total_sequence = PulseSequence()
    readout_map = defaultdict(list)
    for sequence in sequences:
        total_sequence.concatenate(sequence)
        # TODO: Fix unrolling results
        for pulse in sequence.probe_pulses:
            readout_map[pulse.id].append(pulse.id)

        length = sequence.duration + relaxation_time
        for channel in sequence.channels:
            delay = length - sequence.channel_duration(channel)
            total_sequence.append((channel, Delay(duration=delay)))

    return total_sequence, readout_map


def update_configs(configs: dict[str, Config], updates: list[ConfigUpdate]):
    """Apply updates to configs in place.

    Args:
        configs: configs to update. Maps component name to respective config.
        updates: list of config updates. Later entries in the list take precedence over earlier entries
                 (if they happen to update the same thing).
    """
    for update in updates:
        for name, changes in update.items():
            if name not in configs:
                raise ValueError(
                    f"Cannot update configuration for unknown component {name}"
                )
            configs[name] = dataclasses.replace(configs[name], **changes)


def estimate_duration(
    sequences: list[PulseSequence],
    options: ExecutionParameters,
    sweepers: list[ParallelSweepers],
) -> float:
    """Estimate experiment duration."""
    duration = sum(seq.duration for seq in sequences)
    relaxation = default(options.relaxation_time, 0)
    nshots = default(options.nshots, 0)
    return (
        (duration + len(sequences) * relaxation)
        * nshots
        * NS_TO_SEC
        * prod(len(s[0].values) for s in sweepers)
    )


@dataclass
class Settings:
    """Default execution settings read from the runcard."""

    nshots: int = 1024
    """Default number of repetitions when executing a pulse sequence."""
    relaxation_time: int = int(1e5)
    """Time in ns to wait for the qubit to relax to its ground state between
    shots."""

    def fill(self, options: ExecutionParameters):
        """Use default values for missing execution options."""
        if options.nshots is None:
            options = replace(options, nshots=self.nshots)

        if options.relaxation_time is None:
            options = replace(options, relaxation_time=self.relaxation_time)

        return options


@dataclass
class Platform:
    """Platform for controlling quantum devices."""

    name: str
    """Name of the platform."""
    qubits: QubitMap
    """Mapping qubit names to :class:`qibolab.qubits.Qubit` objects."""
    pairs: QubitPairMap
    """Mapping tuples of qubit names to :class:`qibolab.qubits.QubitPair`
    objects."""
    configs: dict[str, Config]
    """Mapping name of component to its default config."""
    instruments: InstrumentMap
    """Mapping instrument names to
    :class:`qibolab.instruments.abstract.Instrument` objects."""

    settings: Settings = field(default_factory=Settings)
    """Container with default execution settings."""
    resonator_type: Optional[str] = None
    """Type of resonator (2D or 3D) in the used QPU.

    Default is 3D for single-qubit chips and 2D for multi-qubit.
    """

    couplers: CouplerMap = field(default_factory=dict)
    """Mapping coupler names to :class:`qibolab.qubits.Qubit` objects."""

    is_connected: bool = False
    """Flag for whether we are connected to the physical instruments."""

    def __post_init__(self):
        log.info("Loading platform %s", self.name)
        if self.resonator_type is None:
            self.resonator_type = "3D" if self.nqubits == 1 else "2D"

    def __str__(self):
        return self.name

    @property
    def nqubits(self) -> int:
        """Total number of usable qubits in the QPU."""
        return len(self.qubits)

    @property
    def ordered_pairs(self):
        """List of qubit pairs that are connected in the QPU."""
        return sorted({tuple(sorted(pair)) for pair in self.pairs})

    @property
    def topology(self) -> list[QubitPairId]:
        """Graph representing the qubit connectivity in the quantum chip."""
        return list(self.pairs)

    @property
    def sampling_rate(self):
        """Sampling rate of control electronics in giga samples per second
        (GSps)."""
        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                return instrument.sampling_rate

    @property
    def components(self) -> set[str]:
        """Names of all components available in the platform."""
        return set(self.configs.keys())

    @property
    def elements(self) -> dict:
        """Qubits and couplers."""
        return self.qubits | self.couplers

    @property
    def channels(self) -> list[str]:
        """Channels in the platform."""
        return list(self.channels_map)

    @property
    def channels_map(self) -> dict[str, QubitId]:
        """Channel to element map."""
        return {ch.name: id for id, el in self.elements.items() for ch in el.channels}

    def config(self, name: str) -> Config:
        """Returns configuration of given component."""
        return self.configs[name]

    def connect(self):
        """Connect to all instruments."""
        if not self.is_connected:
            for instrument in self.instruments.values():
                try:
                    log.info(f"Connecting to instrument {instrument}.")
                    instrument.connect()
                except Exception as exception:
                    raise_error(
                        RuntimeError,
                        f"Cannot establish connection to {instrument} instruments. Error captured: '{exception}'",
                    )
        self.is_connected = True

    def disconnect(self):
        """Disconnects from instruments."""
        if self.is_connected:
            for instrument in self.instruments.values():
                instrument.disconnect()
        self.is_connected = False

    @property
    def _controller(self):
        """Identify controller instrument.

        Used for splitting the unrolled sequences to batches.

        This method does not support platforms with more than one
        controller instruments.
        """
        controllers = [
            instr
            for instr in self.instruments.values()
            if isinstance(instr, Controller)
        ]
        assert len(controllers) == 1
        return controllers[0]

    def _execute(
        self,
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        configs: dict[str, Config],
        integration_setup: IntegrationSetup,
        sweepers: list[ParallelSweepers],
    ):
        """Execute sequences on the controllers."""
        result = {}

        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                new_result = instrument.play(
                    configs, sequences, options, integration_setup, sweepers
                )
                if isinstance(new_result, dict):
                    result.update(new_result)

        return result

    def execute(
        self,
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: Optional[list[ParallelSweepers]] = None,
    ) -> dict[Any, list]:
        """Execute pulse sequences.

        If any sweeper is passed, the execution is performed for the different values
        of sweeped parameters.

        Returns readout results acquired by after execution.

        Example:
            .. testcode::

                import numpy as np
                from qibolab.dummy import create_dummy
                from qibolab.sweeper import Sweeper, Parameter
                from qibolab.pulses import PulseSequence
                from qibolab.execution_parameters import ExecutionParameters


                platform = create_dummy()
                qubit = platform.qubits[0]
                sequence = qubit.native_gates.MZ.create_sequence()
                parameter = Parameter.frequency
                parameter_range = np.random.randint(10, size=10)
                sweeper = [Sweeper(parameter, parameter_range, channels=[qubit.probe.name])]
                platform.execute([sequence], ExecutionParameters(), [sweeper])
        """
        if sweepers is None:
            sweepers = []

        options = self.settings.fill(options)

        time = estimate_duration(sequences, options, sweepers)
        log.info(f"Minimal execution time: {time}")

        configs = self.configs.copy()
        update_configs(configs, options.updates)

        # for components that represent aux external instruments (e.g. lo) to the main control instrument
        # set the config directly
        for name, cfg in configs.items():
            if name in self.instruments:
                self.instruments[name].setup(**asdict(cfg))

        # maps acquisition channel name to corresponding kernel and iq_angle
        # FIXME: this is temporary solution to deliver the information to drivers
        # until we make acquisition channels first class citizens in the sequences
        # so that each acquisition command carries the info with it.
        integration_setup: IntegrationSetup = {}
        for qubit in self.qubits.values():
            integration_setup[qubit.acquisition.name] = qubit.kernel

        results = defaultdict(list)
        for b in batch(sequences, self._controller.bounds):
            result = self._execute(b, options, configs, integration_setup, sweepers)
            for serial, data in result.items():
                results[serial].append(data)

        return results

    def get_qubit(self, qubit):
        """Return the name of the physical qubit corresponding to a logical
        qubit.

        Temporary fix for the compiler to work for platforms where the
        qubits are not named as 0, 1, 2, ...
        """
        try:
            return self.qubits[qubit]
        except KeyError:
            return list(self.qubits.values())[qubit]

    def get_coupler(self, coupler):
        """Return the name of the physical coupler corresponding to a logical
        coupler.

        Temporary fix for the compiler to work for platforms where the
        couplers are not named as 0, 1, 2, ...
        """
        try:
            return self.couplers[coupler]
        except KeyError:
            return list(self.couplers.values())[coupler]
