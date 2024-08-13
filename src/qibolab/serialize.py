"""Helper methods for (de)serializing parameters.

The format is explained in the :ref:`Using parameters <using_runcards>`
example.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import TypeAdapter

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
    single_qubit: dict[QubitId, SingleQubitNatives]
    coupler: dict[QubitId, SingleQubitNatives]
    two_qubit: dict[QubitPairId, TwoQubitNatives]


@dataclass
class Parameters:
    """Serializable parameters."""

    settings: Settings = field(default_factory=Settings)
    configs: dict[str, Config] = field(default_factory=dict)
    # TODO: add gates template
    native_gates: NativeGates = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path):
        """Load parameters from JSON."""
        d = json.loads((path / PARAMETERS).read_text())
        settings = Settings.model_validate(d["settings"])
        configs = TypeAdapter(dict[str, Config]).validate_python(d["components"])
        natives = NativeGates.model_validate(d["native_gates"])
        return cls(settings=settings, configs=configs, native_gates=natives)

    def dump(self, path: Path, updates: Optional[list[ConfigUpdate]] = None):
        """Platform serialization as parameters (json) and kernels (npz).

        The file saved follows the format explained in :ref:`Using parameters <using_runcards>`.

        The requested ``path`` is the folder where the json and npz will be dumped.

        ``updates`` is an optional list if updates for platform configs. Later entries in the list take precedence over earlier ones (if they happen to update the same thing).
        """
        configs = self.configs.copy()
        update_configs(configs, updates or [])

        settings = {
            "settings": self.settings.model_dump(),
            "components": TypeAdapter(dict[str, Config]).dump_python(configs),
            "native_gates": self.native_gates.dump(),
        }

        (path / PARAMETERS).write_text(json.dumps(settings, sort_keys=False, indent=4))


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
