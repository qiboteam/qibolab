"""A platform for executing quantum algorithms."""

from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Literal, Optional, TypeVar

from qibo.config import log, raise_error

from ..components import Config
from ..components.channels import Channel
from ..execution_parameters import ExecutionParameters
from ..identifier import ChannelId, QubitId, QubitPairId, Result
from ..instruments.abstract import Controller, Instrument, InstrumentId
from ..parameters import NativeGates, Parameters, Settings, update_configs
from ..pulses import PulseId
from ..qubits import Qubit
from ..sequence import PulseSequence
from ..sweeper import ParallelSweepers
from ..unrolling import Bounds, batch

__all__ = ["Platform"]

QubitMap = dict[QubitId, Qubit]
QubitPairMap = list[QubitPairId]
InstrumentMap = dict[InstrumentId, Instrument]

NS_TO_SEC = 1e-9
PARAMETERS = "parameters.json"

# TODO: replace with https://docs.python.org/3/reference/compound_stmts.html#type-params
T = TypeVar("T")


# TODO: lift for general usage in Qibolab
def default(value: Optional[T], default: T) -> T:
    """None replacement shortcut."""
    return value if value is not None else default


def _channels_map(elements: QubitMap) -> dict[ChannelId, QubitId]:
    """Map channel names to element (qubit or coupler)."""
    return {ch: id for id, el in elements.items() for ch in el.channels}


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


def _unique_acquisitions(sequences: list[PulseSequence]) -> bool:
    """Check unique acquisition identifiers."""
    ids = []
    for seq in sequences:
        ids += (p.id for _, p in seq.acquisitions)

    return len(ids) == len(set(ids))


@dataclass
class Platform:
    """Platform for controlling quantum devices."""

    name: str
    """Name of the platform."""
    parameters: Parameters
    """..."""
    instruments: InstrumentMap
    """Mapping instrument names to
    :class:`qibolab.instruments.abstract.Instrument` objects."""
    qubits: QubitMap
    """Qubit controllers.

    The mapped objects hold the :class:`qubit.components.channels.Channel` instances
    required to send pulses addressing the desired qubits.
    """
    couplers: QubitMap = field(default_factory=dict)
    """Coupler controllers.

    Fully analogue to :attr:`qubits`. Only the flux channel is expected to be populated
    in the mapped objects.
    """
    resonator_type: Literal["2D", "3D"] = "2D"
    """Type of resonator (2D or 3D) in the used QPU."""
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
    def pairs(self) -> list[QubitPairId]:
        """Available pairs in thee platform."""
        return list(self.parameters.native_gates.two_qubit)

    @property
    def ordered_pairs(self):
        """List of qubit pairs that are connected in the QPU."""
        return sorted({tuple(sorted(pair)) for pair in self.pairs})

    @property
    def settings(self) -> Settings:
        """Container with default execution settings."""
        return self.parameters.settings

    @property
    def natives(self) -> NativeGates:
        """Native gates containers."""
        return self.parameters.native_gates

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
        return set(self.parameters.configs.keys())

    @property
    def channels(self) -> dict[ChannelId, Channel]:
        """Channels in the platform."""
        return {
            id: ch
            for instr in self.instruments.values()
            if isinstance(instr, Controller)
            for id, ch in instr.channels.items()
        }

    @property
    def qubit_channels(self) -> dict[ChannelId, QubitId]:
        """Channel to qubit map."""
        return _channels_map(self.qubits)

    @property
    def coupler_channels(self):
        """Channel to coupler map."""
        return _channels_map(self.couplers)

    def config(self, name: str) -> Config:
        """Returns configuration of given component."""
        # pylint: disable=unsubscriptable-object
        return self.parameters.configs[name]

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
        sweepers: list[ParallelSweepers],
    ) -> dict[PulseId, Result]:
        """Execute sequences on the controllers."""
        result = {}

        for instrument in self.instruments.values():
            if isinstance(instrument, Controller):
                new_result = instrument.play(configs, sequences, options, sweepers)
                if isinstance(new_result, dict):
                    result.update(new_result)

        return result

    def execute(
        self,
        sequences: list[PulseSequence],
        sweepers: Optional[list[ParallelSweepers]] = None,
        **options,
    ) -> dict[PulseId, Result]:
        """Execute pulse sequences.

        If any sweeper is passed, the execution is performed for the different values
        of sweeped parameters.

        Returns readout results acquired by after execution.

        Example:
            .. testcode::

                import numpy as np
                from qibolab import ExecutionParameters, Parameter, PulseSequence, Sweeper, create_dummy


                platform = create_dummy()
                qubit = platform.qubits[0]
                natives = platform.natives.single_qubit[0]
                sequence = natives.MZ.create_sequence()
                parameter_range = np.random.randint(10, size=10)
                sweeper = [
                    Sweeper(
                        parameter=Parameter.frequency,
                        values=parameter_range,
                        channels=[qubit.probe],
                    )
                ]
                platform.execute([sequence], ExecutionParameters(), [sweeper])
        """
        if sweepers is None:
            sweepers = []
        if not _unique_acquisitions(sequences):
            raise ValueError(
                "The acquisitions' identifiers have to be unique across all sequences."
            )

        options = self.settings.fill(ExecutionParameters(**options))

        time = estimate_duration(sequences, options, sweepers)
        log.info(f"Minimal execution time: {time}")

        configs = self.parameters.configs.copy()
        update_configs(configs, options.updates)

        # for components that represent aux external instruments (e.g. lo) to the main
        # control instrument set the config directly
        for name, cfg in configs.items():
            if name in self.instruments:
                self.instruments[name].setup(**cfg.model_dump(exclude={"kind"}))

        results = {}
        # pylint: disable=unsubscriptable-object
        bounds = self.parameters.configs[self._controller.bounds]
        assert isinstance(bounds, Bounds)
        for b in batch(sequences, bounds):
            results |= self._execute(b, options, configs, sweepers)

        return results

    @classmethod
    def load(
        cls,
        path: Path,
        instruments: InstrumentMap,
        qubits: QubitMap,
        couplers: Optional[QubitMap] = None,
        name: Optional[str] = None,
    ) -> "Platform":
        """Dump platform."""
        if name is None:
            name = path.name
        if couplers is None:
            couplers = {}

        return cls(
            name=name,
            parameters=Parameters.model_validate_json((path / PARAMETERS).read_text()),
            instruments=instruments,
            qubits=qubits,
            couplers=couplers,
        )

    def dump(self, path: Path):
        """Dump platform."""
        (path / PARAMETERS).write_text(self.parameters.model_dump_json(indent=4))

    def _element(self, qubit: QubitId, coupler=False) -> tuple[QubitId, Qubit]:
        elements = self.qubits if not coupler else self.couplers
        try:
            return qubit, elements[qubit]
        except KeyError:
            assert isinstance(qubit, int)
            return list(self.qubits.items())[qubit]

    def qubit(self, qubit: QubitId) -> tuple[QubitId, Qubit]:
        """Retrieve physical qubit name and object.

        Temporary fix for the compiler to work for platforms where the
        qubits are not named as 0, 1, 2, ...
        """
        return self._element(qubit)

    def coupler(self, coupler: QubitId) -> tuple[QubitId, Qubit]:
        """Retrieve physical coupler name and object.

        Temporary fix for the compiler to work for platforms where the
        couplers are not named as 0, 1, 2, ...
        """
        return self._element(coupler, coupler=True)
