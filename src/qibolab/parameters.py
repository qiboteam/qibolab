"""Helper methods for (de)serializing parameters.

The format is explained in the :ref:`Using parameters <using_runcards>`
example.
"""

from collections.abc import Callable
from typing import Any

from pydantic import Field, TypeAdapter
from pydantic_core import core_schema

from qibolab.components import Config
from qibolab.execution_parameters import ConfigUpdate, ExecutionParameters
from qibolab.native import SingleQubitNatives, TwoQubitNatives
from qibolab.qubits import QubitId, QubitPairId
from qibolab.serialize import Model, replace


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


class Parameters(Model):
    """Serializable parameters."""

    settings: Settings = Field(default_factory=Settings)
    configs: dict[ComponentId, Config] = Field(default_factory=dict)
    native_gates: NativeGates = Field(default_factory=NativeGates)
