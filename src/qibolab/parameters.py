"""Helper methods for (de)serializing parameters.

The format is explained in the :ref:`Using parameters <using_runcards>`
example.
"""

from collections.abc import Iterable

from pydantic import Field, TypeAdapter, model_serializer, model_validator

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


class TwoQubitContainer(Model):
    pairs: dict[QubitPairId, TwoQubitNatives] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def wrap(cls, data: dict) -> dict:
        return {"pairs": data}

    @model_serializer
    def unwrap(self) -> dict:
        return TypeAdapter(dict[QubitPairId, TwoQubitNatives]).dump_python(self.pairs)

    def __getitem__(self, key: QubitPairId):
        try:
            return self.pairs[key]
        except KeyError as e:
            value = self.pairs[(key[1], key[0])]
            if value.symmetric:
                return value
            raise e

    def __setitem__(self, key: QubitPairId, value: TwoQubitNatives):
        self.pairs[key] = value

    def __delitem__(self, key: QubitPairId):
        del self.pairs[key]

    def __contains__(self, key: QubitPairId) -> bool:
        return key in self.pairs

    def __iter__(self) -> Iterable[QubitPairId]:
        return iter(self.pairs)

    def keys(self) -> Iterable[QubitPairId]:
        return self.pairs.keys()

    def values(self) -> Iterable[TwoQubitNatives]:
        return self.pairs.values()

    def items(self) -> Iterable[tuple[QubitPairId, TwoQubitNatives]]:
        return self.pairs.items()


class NativeGates(Model):
    """Native gates parameters.

    This is a container for the parameters of the whole platform.
    """

    single_qubit: dict[QubitId, SingleQubitNatives] = Field(default_factory=dict)
    coupler: dict[QubitId, SingleQubitNatives] = Field(default_factory=dict)
    two_qubit: TwoQubitContainer = Field(default_factory=TwoQubitContainer)


ComponentId = str


class Parameters(Model):
    """Serializable parameters."""

    settings: Settings = Field(default_factory=Settings)
    configs: dict[ComponentId, Config] = Field(default_factory=dict)
    native_gates: NativeGates = Field(default_factory=NativeGates)
