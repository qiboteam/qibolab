"""Helper methods for (de)serializing parameters.

The format is explained in the :ref:`Loading platform parameters from
JSON <parameters_json>` example.
"""

from collections.abc import Callable, Iterable
from typing import Annotated, Any, Union

from pydantic import BeforeValidator, Field, PlainSerializer, TypeAdapter
from pydantic_core import core_schema

from .components import ChannelConfig, Config
from .execution_parameters import ConfigUpdate, ExecutionParameters
from .identifier import QubitId, QubitPairId
from .native import SingleQubitNatives, TwoQubitNatives
from .serialize import Model, replace
from .unrolling import Bounds

__all__ = ["ConfigKinds"]


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
    for acc in steps[:-1]:
        current = current[acc]

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

    def update(self, update: Update) -> "Parameters":
        """Update parameters' values."""
        d = self.model_dump()
        for path, val in update.items():
            _setvalue(d, path, val)

        return self.model_validate(d)
