"""Helper methods for (de)serializing parameters.

The format is explained in the :ref:`Loading platform parameters from
JSON <parameters_json>` example.
"""

from collections.abc import Callable, Iterable
from typing import Annotated, Any, Union

from pydantic import BeforeValidator, Field, PlainSerializer, TypeAdapter
from pydantic_core import core_schema
from typing_extensions import NotRequired, TypedDict

from .components import ChannelConfig, Config, channel_to_config
from .execution_parameters import ConfigUpdate, ExecutionParameters
from .identifier import QubitId, QubitPairId
from .instruments.abstract import Instrument, InstrumentId
from .native import Native, NativeContainer, SingleQubitNatives, TwoQubitNatives
from .pulses import Acquisition, Pulse, Readout, Rectangular
from .qubits import Qubit
from .serialize import Model, replace
from .unrolling import Bounds

__all__ = ["ConfigKinds", "QubitMap", "InstrumentMap", "Hardware", "ParametersBuilder"]


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
            configs[name] = replace(configs[name], **changes)


class Settings(Model):
    """Default platform execution settings."""

    nshots: int = 1000
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


class TwoQubitContainer(dict[QubitPairId, TwoQubitNatives]):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Callable[[Any], core_schema.CoreSchema]
    ) -> core_schema.CoreSchema:
        schema = handler(dict[QubitPairId, TwoQubitNatives])
        return core_schema.no_info_after_validator_function(
            cls._validate,
            schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize, info_arg=False
            ),
        )

    @classmethod
    def _validate(cls, value):
        return cls(value)

    @staticmethod
    def _serialize(value):
        return TypeAdapter(dict[QubitPairId, TwoQubitNatives]).dump_python(value)

    def __getitem__(self, key: QubitPairId):
        try:
            return super().__getitem__(key)
        except KeyError:
            value = super().__getitem__((key[1], key[0]))
            if value.symmetric:
                return value
            raise


class NativeGates(Model):
    """Native gates parameters.

    This is a container for the parameters of the whole platform.
    """

    single_qubit: dict[QubitId, SingleQubitNatives] = Field(default_factory=dict)
    coupler: dict[QubitId, SingleQubitNatives] = Field(default_factory=dict)
    two_qubit: TwoQubitContainer = Field(default_factory=dict)


ComponentId = str
"""Identifier of a generic component.

This is assumed to always be in its serialized form.
"""

# TODO: replace _UnionType with UnionType, once py3.9 will be abandoned
_UnionType = Any
_ChannelConfigT = Union[_UnionType, type[Config]]
_BUILTIN_CONFIGS: tuple[_ChannelConfigT, ...] = (ChannelConfig, Bounds)


class ConfigKinds:
    """Registered configuration kinds.

    This class is handling the known configuration kinds for deserialization.

    .. attention::

        Beware that is managing a global state. This should not be a major issue, as the
        known configurations should be fixed per run. But prefer avoiding changing them
        during a single session, unless you are clearly controlling the sequence of all
        loading operations.
    """

    _registered: list[_ChannelConfigT] = list(_BUILTIN_CONFIGS)

    @classmethod
    def extend(cls, kinds: Iterable[_ChannelConfigT]):
        """Extend the known configuration kinds.

        Nested unions are supported (i.e. :class:`Union` as elements of ``kinds``).
        """
        cls._registered.extend(kinds)

    @classmethod
    def reset(cls):
        """Reset known configuration kinds to built-ins."""
        cls._registered = list(_BUILTIN_CONFIGS)

    @classmethod
    def registered(cls) -> list[_ChannelConfigT]:
        """Retrieve registered configuration kinds."""
        return cls._registered.copy()

    @classmethod
    def adapted(cls) -> TypeAdapter:
        """Construct tailored pydantic type adapter.

        The adapter will be able to directly load all the registered
        configuration kinds as the appropriate Python objects.
        """
        return TypeAdapter(
            Annotated[
                Union[tuple(ConfigKinds._registered)], Field(discriminator="kind")
            ]
        )


def _load_configs(raw: dict[str, dict]) -> dict[ComponentId, Config]:
    a = ConfigKinds.adapted()
    return {k: a.validate_python(v) for k, v in raw.items()}


def _dump_configs(obj: dict[ComponentId, Config]) -> dict[str, dict]:
    a = ConfigKinds.adapted()
    return {k: a.dump_python(v) for k, v in obj.items()}


def _setvalue(d: dict, path: str, val: Any):
    steps = path.split(".")
    current = d
    for step in steps[:-1]:
        try:
            current = current[int(step)]
        except ValueError:
            current = current[step]

    current[steps[-1]] = val


Update = dict[str, Any]


class Parameters(Model):
    """Serializable parameters."""

    settings: Settings = Field(default_factory=Settings)
    configs: Annotated[
        dict[ComponentId, Config],
        BeforeValidator(_load_configs),
        PlainSerializer(_dump_configs),
    ] = Field(default_factory=dict)
    native_gates: NativeGates = Field(default_factory=NativeGates)

    def replace(self, update: Update) -> "Parameters":
        """Update parameters' values."""
        d = self.model_dump()
        for path, val in update.items():
            _setvalue(d, path, val)

        return self.model_validate(d)


QubitMap = dict[QubitId, Qubit]
InstrumentMap = dict[InstrumentId, Instrument]


class Hardware(TypedDict):
    """Part of the platform that specifies the hardware configuration."""

    instruments: InstrumentMap
    qubits: QubitMap
    couplers: NotRequired[QubitMap]


def _gate_channel(qubit: Qubit, gate: str) -> str:
    """Default channel that a native gate plays on."""
    if gate in ("RX", "RX90", "CNOT"):
        return qubit.drive
    if gate == "RX12":
        return qubit.drive_qudits[(1, 2)]
    if gate == "MZ":
        return qubit.acquisition
    if gate in ("CP", "CZ", "iSWAP"):
        return qubit.flux


def _gate_sequence(qubit: Qubit, gate: str) -> Native:
    """Default sequence corresponding to a native gate."""
    channel = _gate_channel(qubit, gate)
    pulse = Pulse(duration=0, amplitude=0, envelope=Rectangular())
    if gate != "MZ":
        return Native([(channel, pulse)])

    return Native(
        [(channel, Readout(acquisition=Acquisition(duration=0), probe=pulse))]
    )


def _pair_to_qubit(pair: str, qubits: QubitMap) -> Qubit:
    """Get first qubit of a pair given in ``{q0}-{q1}`` format."""
    q = tuple(pair.split("-"))[0]
    try:
        return qubits[q]
    except KeyError:
        return qubits[int(q)]


def _native_builder(cls, qubit: Qubit, natives: set[str]) -> NativeContainer:
    """Build default native gates for a given qubit or pair.

    In case of pair, ``qubit`` is assumed to be the first qubit of the pair,
    and a default pulse is added on that qubit, because at this stage we don't
    know which qubit is the high frequency one.
    """
    return cls(
        **{
            gate: _gate_sequence(qubit, gate)
            for gate in cls.model_fields.keys() & natives
        }
    )


class ParametersBuilder(Model):
    """Generates default ``Parameters`` for a given platform hardware
    configuration."""

    hardware: Hardware
    natives: set[str] = Field(default_factory=set)
    pairs: list[str] = Field(default_factory=list)

    def build(self) -> Parameters:
        settings = Settings()

        configs = {}
        for instrument in self.hardware.get("instruments", {}).values():
            if hasattr(instrument, "channels"):
                configs |= {
                    id: channel_to_config(channel)
                    for id, channel in instrument.channels.items()
                }

        qubits = self.hardware.get("qubits", {})
        single_qubit = {
            q: _native_builder(SingleQubitNatives, qubit, self.natives - {"CP"})
            for q, qubit in qubits.items()
        }
        coupler = {
            q: _native_builder(SingleQubitNatives, qubit, self.natives & {"CP"})
            for q, qubit in self.hardware.get("couplers", {}).items()
        }
        two_qubit = {
            pair: _native_builder(
                TwoQubitNatives, _pair_to_qubit(pair, qubits), self.natives
            )
            for pair in self.pairs
        }
        native_gates = NativeGates(
            single_qubit=single_qubit, coupler=coupler, two_qubit=two_qubit
        )

        return Parameters(settings=settings, configs=configs, native_gates=native_gates)
