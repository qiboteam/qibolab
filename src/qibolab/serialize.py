"""Helper methods for (de)serializing parameters.

The format is explained in the :ref:`Using parameters <using_runcards>`
example.
"""

import json
from pathlib import Path

from pydantic import Field

from qibolab.components import Config
from qibolab.execution_parameters import ConfigUpdate, ExecutionParameters
from qibolab.kernels import Kernels
from qibolab.native import SingleQubitNatives, TwoQubitNatives
from qibolab.qubits import Qubit, QubitId, QubitPair, QubitPairId
from qibolab.serialize_ import Model, replace

PARAMETERS = "parameters.json"
PLATFORM = "platform.py"

QubitMap = dict[QubitId, Qubit]
QubitPairMap = dict[QubitPairId, QubitPair]


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


class NativeGates(Model):
    single_qubit: dict[QubitId, SingleQubitNatives] = Field(default_factory=dict)
    coupler: dict[QubitId, SingleQubitNatives] = Field(default_factory=dict)
    two_qubit: dict[QubitPairId, TwoQubitNatives] = Field(default_factory=dict)


class Parameters(Model):
    """Serializable parameters."""

    settings: Settings = Field(default_factory=Settings)
    configs: dict[str, Config] = Field(default_factory=dict)
    native_gates: NativeGates = Field(default_factory=NativeGates)

    @classmethod
    def load(cls, path: Path):
        """Load parameters from JSON."""
        return cls.model_validate(json.loads((path / PARAMETERS).read_text()))

    def dump(self, path: Path):
        """Platform serialization as parameters (json) and kernels (npz).

        The file saved follows the format explained in :ref:`Using parameters <using_runcards>`.

        The requested ``path`` is the folder where the json and npz will be dumped.
        """
        (path / PARAMETERS).write_text(
            json.dumps(self.model_dump(), sort_keys=False, indent=4)
        )


# TODO: kernels are part of the parameters, they should not be dumped separately
def dump_kernels(platform: "Platform", path: Path):
    """Creates Kernels instance from platform and dumps as npz.

    Args:
        platform (qibolab.platform.Platform): The platform to be serialized.
        path (pathlib.Path): Path that the kernels file will be saved.
    """

    # create kernels
    kernels = Kernels()
    for qubit in platform.qubits.values():
        kernel = platform.configs[qubit.acquisition.name].kernel
        if kernel is not None:
            kernels[qubit.name] = kernel

    # dump only if not None
    if len(kernels) > 0:
        kernels.dump(path)


# TODO: drop as soon as dump_kernels is reabsorbed in the parameters
def dump_platform(platform: "Platform", path: Path):
    """Dump paltform."""
    platform.parameters.dump(path)
    dump_kernels(platform, path)
