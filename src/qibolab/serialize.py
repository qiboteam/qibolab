"""Helper methods for (de)serializing parameters.

The format is explained in the :ref:`Using parameters <using_runcards>`
example.
"""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Annotated, Optional, Union

from pydantic import Field, TypeAdapter

from qibolab.components import Config
from qibolab.execution_parameters import ConfigUpdate, ExecutionParameters
from qibolab.kernels import Kernels
from qibolab.native import FixedSequenceFactory, SingleQubitNatives, TwoQubitNatives
from qibolab.pulses import PulseSequence
from qibolab.pulses.pulse import PulseLike
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


@dataclass
class NativeGates:
    single_qubit: dict[QubitId, Qubit]
    coupler: dict[QubitId, Qubit]
    two_qubit: dict[QubitPairId, QubitPair]

    @classmethod
    def load(cls, raw: dict):
        """Load qubits, couplers and pairs."""
        qubits = _load_single_qubit_natives(raw["single_qubit"])
        couplers = _load_single_qubit_natives(raw["coupler"])
        pairs = _load_two_qubit_natives(raw["two_qubit"], qubits)
        return cls(qubits, couplers, pairs)

    def dump(self) -> dict:
        """Serialize native gates section to dictionary."""
        native_gates = {
            "single_qubit": {
                _dump_qubit_name(q): _dump_natives(qubit.native_gates)
                for q, qubit in self.single_qubit.items()
            }
        }

        native_gates["coupler"] = {
            _dump_qubit_name(q): _dump_natives(qubit.native_gates)
            for q, qubit in self.coupler.items()
        }

        native_gates["two_qubit"] = {}
        for pair in self.two_qubit.values():
            natives = _dump_natives(pair.native_gates)
            if len(natives) > 0:
                pair_name = f"{pair.qubit1}-{pair.qubit2}"
                native_gates["two_qubit"][pair_name] = natives

        return native_gates


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
        settings = Settings(**d["settings"])
        configs = TypeAdapter(dict[str, Config]).validate_python(d["components"])
        natives = NativeGates.load(d["native_gates"])
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


def _load_qubit_name(name: str) -> QubitId:
    """Convert qubit name from string to integer or string."""
    return TypeAdapter(
        Annotated[Union[int, str], Field(union_mode="left_to_right")]
    ).validate_python(name)


def _load_pulse(pulse_kwargs: dict):
    return TypeAdapter(PulseLike).validate_python(pulse_kwargs)


def _load_sequence(raw_sequence):
    return PulseSequence([(ch, _load_pulse(pulse)) for ch, pulse in raw_sequence])


def _load_single_qubit_natives(gates: dict) -> dict[QubitId, Qubit]:
    """Parse native gates."""
    qubits = {}
    for q, gatedict in gates.items():
        name = _load_qubit_name(q)
        native_gates = SingleQubitNatives(**gatedict)
        qubits[name] = Qubit(name=name, native_gates=native_gates)
    return qubits


def _load_two_qubit_natives(
    gates: dict, qubits: dict[QubitId, Qubit]
) -> dict[QubitPairId, QubitPair]:
    pairs = {}
    for pair, gatedict in gates.items():
        q0, q1 = (_load_qubit_name(q) for q in pair.split("-"))
        native_gates = TwoQubitNatives(
            **{
                gate_name: FixedSequenceFactory(_load_sequence(raw_sequence))
                for gate_name, raw_sequence in gatedict.items()
            }
        )
        pairs[(q0, q1)] = QubitPair(q0, q1, native_gates=native_gates)
        if native_gates.symmetric:
            pairs[(q1, q0)] = pairs[(q0, q1)]
    return pairs


def _dump_qubit_name(name: QubitId) -> str:
    """Convert qubit name from integer or string to string."""
    if isinstance(name, int):
        return str(name)
    return name


def _dump_pulse(pulse: PulseLike):
    data = pulse.model_dump()
    if "channel" in data:
        del data["channel"]
    if "relative_phase" in data:
        del data["relative_phase"]
    return data


def _dump_sequence(sequence: PulseSequence):
    return [(ch, _dump_pulse(p)) for ch, p in sequence]


def _dump_natives(natives: Union[SingleQubitNatives, TwoQubitNatives]):
    data = {}
    for fld in fields(natives):
        factory = getattr(natives, fld.name)
        if factory is not None:
            data[fld.name] = _dump_sequence(factory._seq)
    return data


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
