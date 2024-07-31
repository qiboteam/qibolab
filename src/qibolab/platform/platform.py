"""A platform for executing quantum algorithms."""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from math import prod
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import networkx as nx
import numpy as np
from qibo.config import log, raise_error

from qibolab.components import Config
from qibolab.couplers import Coupler
from qibolab.execution_parameters import ExecutionParameters
from qibolab.instruments.abstract import Controller, Instrument, InstrumentId
from qibolab.pulses import Delay, PulseSequence
from qibolab.qubits import Qubit, QubitId, QubitPair, QubitPairId
from qibolab.serialize_ import replace
from qibolab.sweeper import ParallelSweepers
from qibolab.unrolling import batch

InstrumentMap = Dict[InstrumentId, Instrument]
QubitMap = Dict[QubitId, Qubit]
CouplerMap = Dict[QubitId, Coupler]
QubitPairMap = Dict[QubitPairId, QubitPair]

NS_TO_SEC = 1e-9

# TODO: replace with https://docs.python.org/3/reference/compound_stmts.html#type-params
T = TypeVar("T")


# TODO: lift for general usage in Qibolab
def default(value: Optional[T], default: T) -> T:
    """None replacement shortcut."""
    return value if value is not None else default


def unroll_sequences(
    sequences: List[PulseSequence], relaxation_time: int
) -> Tuple[PulseSequence, dict[str, list[str]]]:
    """Unrolls a list of pulse sequences to a single pulse sequence with
    multiple measurements.

    Args:
        sequences (list): List of pulse sequences to unroll.
        relaxation_time (int): Time in ns to wait for the qubit to relax between
            playing different sequences.

    Returns:
        total_sequence (:class:`qibolab.pulses.PulseSequence`): Unrolled pulse sequence containing
            multiple measurements.
        readout_map (dict): Map from original readout pulse serials to the unrolled readout pulse
            serials. Required to construct the results dictionary that is returned after execution.
    """
    total_sequence = PulseSequence()
    readout_map = defaultdict(list)
    for sequence in sequences:
        total_sequence.extend(sequence)
        # TODO: Fix unrolling results
        for pulse in sequence.probe_pulses:
            readout_map[pulse.id].append(pulse.id)

        length = sequence.duration + relaxation_time
        for channel in sequence.keys():
            delay = length - sequence.channel_duration(channel)
            total_sequence[channel].append(Delay(duration=delay))

    return total_sequence, readout_map


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
    """Dictionary mapping qubit names to :class:`qibolab.qubits.Qubit`
    objects."""
    pairs: QubitPairMap
    """Dictionary mapping tuples of qubit names to
    :class:`qibolab.qubits.QubitPair` objects."""
    configs: dict[str, Config]
    """Maps name of component to its default config."""
    instruments: InstrumentMap
    """Dictionary mapping instrument names to
    :class:`qibolab.instruments.abstract.Instrument` objects."""

    settings: Settings = field(default_factory=Settings)
    """Container with default execution settings."""
    resonator_type: Optional[str] = None
    """Type of resonator (2D or 3D) in the used QPU.

    Default is 3D for single-qubit chips and 2D for multi-qubit.
    """

    couplers: CouplerMap = field(default_factory=dict)
    """Dictionary mapping coupler names to :class:`qibolab.couplers.Coupler`
    objects."""

    is_connected: bool = False
    """Flag for whether we are connected to the physical instruments."""

    topology: nx.Graph = field(default_factory=nx.Graph)
    """Graph representing the qubit connectivity in the quantum chip."""

    def __post_init__(self):
        log.info("Loading platform %s", self.name)
        if self.resonator_type is None:
            self.resonator_type = "3D" if self.nqubits == 1 else "2D"

        self.topology.add_nodes_from(self.qubits.keys())
        self.topology.add_edges_from(
            [(pair.qubit1.name, pair.qubit2.name) for pair in self.pairs.values()]
        )

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

    def _apply_config_updates(
        self, updates: list[dict[str, Config]]
    ) -> dict[str, Config]:
        """Apply given list of config updates to the default configuration and
        return the updated config dict.

        Args:
            updates: list of updates, where each entry is a dict mapping component name to new config. Later entries
                     in the list override earlier entries (if they happen to update the same thing).
        """
        components = self.configs.copy()
        for update in updates:
            for name, cfg in update.items():
                if name not in components:
                    raise ValueError(
                        f"Cannot update configuration for unknown component {name}"
                    )
                if type(cfg) is not type(components[name]):
                    raise ValueError(
                        f"Configuration of component {name} with type {type(components[name])} cannot be updated with configuration of type {type(cfg)}"
                    )
                components[name] = cfg
        return components

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

    def _execute(self, sequences, options, integration_setup, sweepers):
        """Execute sequence on the controllers."""
        result = {}

        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                new_result = instrument.play(
                    options.configs, sequences, options, integration_setup, sweepers
                )
                if isinstance(new_result, dict):
                    result.update(new_result)

        return result

    def execute(
        self,
        sequences: List[PulseSequence],
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
                sweeper = [Sweeper(parameter, parameter_range, channels=[qubit.measure.name])]
                platform.execute([sequence], ExecutionParameters(), [sweeper])
        """
        if sweepers is None:
            sweepers = []

        options = self.settings.fill(options)

        time = estimate_duration(sequences, options, sweepers)
        log.info(f"Minimal execution time: {time}")

        configs = self._apply_config_updates(options.configs)

        # for components that represent aux external instruments (e.g. lo) to the main control instrument
        # set the config directly
        for name, cfg in configs.items():
            if name in self.instruments:
                self.instruments[name].setup(**asdict(cfg))

        # maps acquisition channel name to corresponding kernel and iq_angle
        # FIXME: this is temporary solution to deliver the information to drivers
        # until we make acquisition channels first class citizens in the sequences
        # so that each acquisition command carries the info with it.
        integration_setup: dict[str, tuple[np.ndarray, float]] = {}
        for qubit in self.qubits.values():
            integration_setup[qubit.acquisition.name] = (qubit.kernel, qubit.iq_angle)

        # find readout pulses
        ro_pulses = {
            pulse.id: pulse.qubit
            for sequence in sequences
            for pulse in sequence.ro_pulses
        }

        results = defaultdict(list)
        for b in batch(sequences, self._controller.bounds):
            result = self._execute(b, options, integration_setup, sweepers)
            for serial, data in result.items():
                results[serial].append(data)

        for serial, qubit in ro_pulses.items():
            results[qubit] = results[serial]

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
