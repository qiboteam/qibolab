"""Helper methods for loading and saving to runcards.

The format of runcards in the ``qiboteam/qibolab_platforms_qrc``
repository is assumed here. See :ref:`Using runcards <using_runcards>`
example for more details.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Annotated, Optional, Union

from pydantic import Field, TypeAdapter

from qibolab.components import AcquisitionConfig
from qibolab.execution_parameters import ConfigUpdate
from qibolab.kernels import Kernels
from qibolab.native import (
    FixedSequenceFactory,
    RxyFactory,
    SingleQubitNatives,
    TwoQubitNatives,
)
from qibolab.platform.platform import (
    InstrumentMap,
    Platform,
    QubitMap,
    QubitPairMap,
    Settings,
    update_configs,
)
from qibolab.pulses import PulseSequence
from qibolab.pulses.pulse import PulseLike
from qibolab.qubits import Qubit, QubitId, QubitPair, QubitPairId

RUNCARD = "parameters.json"
PLATFORM = "platform.py"


@dataclass
class NativeGates:
    single_qubit: dict[QubitId, Qubit]
    coupler: dict[QubitId, Qubit]
    two_qubit: dict[QubitPairId, QubitPair]

    @classmethod
    def load(cls, raw: dict):
        """Load qubits, couplers and pairs from the runcard.

        Uses the native gate section of the runcard to parse the
        corresponding :class: `qibolab.qubits.Qubit` and
        :class: `qibolab.qubits.QubitPair` objects.
        """
        qubits = _load_single_qubit_natives(raw["single_qubit"])
        couplers = _load_single_qubit_natives(raw["coupler"])
        pairs = _load_two_qubit_natives(raw["two_qubit"], qubits)
        return cls(qubits, couplers, pairs)


@dataclass
class Runcard:
    settings: Settings
    components: dict
    native_gates: NativeGates

    @classmethod
    def load(cls, path: Path):
        """Load runcard from JSON."""
        d = json.loads((path / RUNCARD).read_text())
        settings = Settings(**d["settings"])
        components = d["components"]
        natives = NativeGates.load(d["native_gates"])
        return cls(settings=settings, components=components, native_gates=natives)


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
    """Parse native gates from the runcard.

    Args:
        gates (dict): Dictionary with native gate pulse parameters as loaded
            from the runcard.
    """
    qubits = {}
    for q, gatedict in gates.items():
        name = _load_qubit_name(q)
        native_gates = SingleQubitNatives(
            **{
                gate_name: (
                    RxyFactory(_load_sequence(raw_sequence))
                    if gate_name == "RX"
                    else FixedSequenceFactory(_load_sequence(raw_sequence))
                )
                for gate_name, raw_sequence in gatedict.items()
            }
        )
        qubits[name] = Qubit(_load_qubit_name(q), native_gates=native_gates)
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


def dump_qubit_name(name: QubitId) -> str:
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


def dump_native_gates(
    qubits: QubitMap, pairs: QubitPairMap, couplers: Optional[QubitMap] = None
) -> dict:
    """Dump native gates section to dictionary following the runcard format,
    using qubit and pair objects."""
    # single-qubit native gates
    native_gates = {
        "single_qubit": {
            dump_qubit_name(q): _dump_natives(qubit.native_gates)
            for q, qubit in qubits.items()
        }
    }

    # couplers native gates
    native_gates["coupler"] = {
        dump_qubit_name(q): _dump_natives(qubit.native_gates)
        for q, qubit in qubits.items()
    }

    # two-qubit native gates
    native_gates["two_qubit"] = {}
    for pair in pairs.values():
        natives = _dump_natives(pair.native_gates)
        if len(natives) > 0:
            pair_name = f"{pair.qubit1}-{pair.qubit2}"
            native_gates["two_qubit"][pair_name] = natives

    return native_gates


def dump_instruments(instruments: InstrumentMap) -> dict:
    """Dump instrument settings to a dictionary following the runcard
    format."""
    # Qblox modules settings are dictionaries and not dataclasses
    data = {}
    for name, instrument in instruments.items():
        try:
            # TODO: Migrate all instruments to this approach
            # (I think it is also useful for qblox)
            settings = instrument.dump()
            if len(settings) > 0:
                data[name] = settings
        except AttributeError:
            settings = instrument.settings
            if settings is not None:
                if isinstance(settings, dict):
                    data[name] = settings
                else:
                    data[name] = settings.dump()

    return data


def dump_component_configs(component_configs) -> dict:
    """Dump channel configs."""
    components = {}
    for name, cfg in component_configs.items():
        components[name] = asdict(cfg)
        if isinstance(cfg, AcquisitionConfig):
            del components[name]["kernel"]
    return components


def dump_runcard(
    platform: Platform, path: Path, updates: Optional[list[ConfigUpdate]] = None
):
    """Serializes the platform and saves it as a json runcard file.

    The file saved follows the format explained in :ref:`Using runcards <using_runcards>`.

    Args:
        platform (qibolab.platform.Platform): The platform to be serialized.
        path (pathlib.Path): Path that the json file will be saved.
        updates: List if updates for platform configs.
                 Later entries in the list take precedence over earlier ones (if they happen to update the same thing).
    """

    configs = platform.configs.copy()
    update_configs(configs, updates or [])

    settings = {
        "settings": asdict(platform.settings),
        "qubits": list(platform.qubits),
        "components": dump_component_configs(configs),
    }

    settings["native_gates"] = dump_native_gates(
        platform.qubits, platform.pairs, platform.couplers
    )

    (path / RUNCARD).write_text(json.dumps(settings, sort_keys=False, indent=4))


def dump_kernels(platform: Platform, path: Path):
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


def dump_platform(
    platform: Platform, path: Path, updates: Optional[list[ConfigUpdate]] = None
):
    """Platform serialization as runcard (json) and kernels (npz).

    Args:
        platform (qibolab.platform.Platform): The platform to be serialized.
        path (pathlib.Path): Path where json and npz will be dumped.
        updates: List if updates for platform configs.
                 Later entries in the list take precedence over earlier ones (if they happen to update the same thing).
    """

    dump_kernels(platform=platform, path=path)
    dump_runcard(platform=platform, path=path, updates=updates)
