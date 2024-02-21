"""Helper methods for loading and saving to runcards.

The format of runcards in the ``qiboteam/qibolab_platforms_qrc``
repository is assumed here. See :ref:`Using runcards <using_runcards>`
example for more details.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

from qibolab.couplers import Coupler
from qibolab.kernels import Kernels
from qibolab.native import CouplerNatives, SingleQubitNatives, TwoQubitNatives
from qibolab.platform import (
    CouplerMap,
    InstrumentMap,
    Platform,
    QubitMap,
    QubitPairMap,
    Settings,
)
from qibolab.qubits import Qubit, QubitPair

RUNCARD = "parameters.json"
PLATFORM = "platform.py"


def load_runcard(path: Path) -> dict:
    """Load runcard JSON to a dictionary."""
    return json.loads((path / RUNCARD).read_text())


def load_settings(runcard: dict) -> Settings:
    """Load platform settings section from the runcard."""
    return Settings(**runcard["settings"])


def load_qubits(
    runcard: dict, kernels: Kernels = None
) -> Tuple[QubitMap, CouplerMap, QubitPairMap]:
    """Load qubits and pairs from the runcard.

    Uses the native gate and characterization sections of the runcard to
    parse the
    :class: `qibolab.qubits.Qubit` and
    :class: `qibolab.qubits.QubitPair`
    objects.
    """
    qubits = {
        json.loads(q): Qubit(json.loads(q), **char)
        for q, char in runcard["characterization"]["single_qubit"].items()
    }
    if kernels is not None:
        for q in kernels:
            qubits[q].kernel = kernels[q]

    couplers = {}
    pairs = {}
    if "coupler" in runcard["characterization"]:
        couplers = {
            json.loads(c): Coupler(json.loads(c), **char)
            for c, char in runcard["characterization"]["coupler"].items()
        }

        for c, pair in runcard["topology"].items():
            q0, q1 = pair
            pairs[(q0, q1)] = pairs[(q1, q0)] = QubitPair(
                qubits[q0], qubits[q1], couplers[json.loads(c)]
            )
    else:
        for pair in runcard["topology"]:
            q0, q1 = pair
            pairs[(q0, q1)] = pairs[(q1, q0)] = QubitPair(qubits[q0], qubits[q1], None)

    qubits, pairs, couplers = register_gates(runcard, qubits, pairs, couplers)

    return qubits, couplers, pairs


# This creates the compiler error
def register_gates(
    runcard: dict, qubits: QubitMap, pairs: QubitPairMap, couplers: CouplerMap = None
) -> Tuple[QubitMap, QubitPairMap]:
    """Register single qubit native gates to ``Qubit`` objects from the
    runcard.

    Uses the native gate and characterization sections of the runcard
    to parse the :class:`qibolab.qubits.Qubit` and :class:`qibolab.qubits.QubitPair`
    objects.
    """

    native_gates = runcard.get("native_gates", {})
    for q, gates in native_gates.get("single_qubit", {}).items():
        qubits[json.loads(q)].native_gates = SingleQubitNatives.from_dict(
            qubits[json.loads(q)], gates
        )

    for c, gates in native_gates.get("coupler", {}).items():
        couplers[json.loads(c)].native_pulse = CouplerNatives.from_dict(
            couplers[json.loads(c)], gates
        )

    # register two-qubit native gates to ``QubitPair`` objects
    for pair, gatedict in native_gates.get("two_qubit", {}).items():
        q0, q1 = tuple(int(q) if q.isdigit() else q for q in pair.split("-"))
        native_gates = TwoQubitNatives.from_dict(qubits, couplers, gatedict)
        coupler = pairs[(q0, q1)].coupler
        pairs[(q0, q1)] = QubitPair(qubits[q0], qubits[q1], coupler, native_gates)
        if native_gates.symmetric:
            pairs[(q1, q0)] = pairs[(q0, q1)]

    return qubits, pairs, couplers


def load_instrument_settings(
    runcard: dict, instruments: InstrumentMap
) -> InstrumentMap:
    """Setup instruments according to the settings given in the runcard."""
    for name, settings in runcard.get("instruments", {}).items():
        instruments[name].setup(**settings)
    return instruments


def dump_native_gates(
    qubits: QubitMap, pairs: QubitPairMap, couplers: CouplerMap = None
) -> dict:
    """Dump native gates section to dictionary following the runcard format,
    using qubit and pair objects."""
    # single-qubit native gates
    native_gates = {
        "single_qubit": {
            json.dumps(q): qubit.native_gates.raw for q, qubit in qubits.items()
        }
    }
    if couplers:
        native_gates["coupler"] = {
            json.dumps(c): coupler.native_pulse.raw for c, coupler in couplers.items()
        }

    # two-qubit native gates
    native_gates["two_qubit"] = {}
    for pair in pairs.values():
        natives = pair.native_gates.raw
        if len(natives) > 0:
            pair_name = f"{pair.qubit1.name}-{pair.qubit2.name}"
            native_gates["two_qubit"][pair_name] = natives

    return native_gates


def dump_characterization(qubits: QubitMap, couplers: CouplerMap = None) -> dict:
    """Dump qubit characterization section to dictionary following the runcard
    format, using qubit and pair objects."""
    characterization = {
        "single_qubit": {
            json.dumps(q): qubit.characterization for q, qubit in qubits.items()
        },
    }

    if couplers:
        characterization["coupler"] = {
            json.dumps(c.name): {"sweetspot": c.sweetspot} for c in couplers.values()
        }
    return characterization


def dump_instruments(instruments: InstrumentMap) -> dict:
    """Dump instrument settings to a dictionary following the runcard
    format."""
    # Qblox modules settings are dictionaries and not dataclasses
    data = {}
    for name, instrument in instruments.items():
        try:
            settings = instrument.settings
            if settings is not None:
                if isinstance(settings, dict):
                    data[name] = settings
                else:
                    data[name] = settings.dump()
        except AttributeError:
            # TODO: Migrate all instruments to this approach
            # (I think it is also useful for qblox)
            settings = instrument.dump()
            if len(settings) > 0:
                data[name] = settings
    return data


def dump_runcard(platform: Platform, path: Path):
    """Serializes the platform and saves it as a json runcard file.

    The file saved follows the format explained in :ref:`Using runcards <using_runcards>`.

    Args:
        platform (qibolab.platform.Platform): The platform to be serialized.
        path (pathlib.Path): Path that the json file will be saved.
    """

    settings = {
        "nqubits": platform.nqubits,
        "settings": asdict(platform.settings),
        "qubits": list(platform.qubits),
        "topology": [list(pair) for pair in platform.ordered_pairs],
        "instruments": dump_instruments(platform.instruments),
    }

    if platform.couplers:
        settings["couplers"] = list(platform.couplers)
        settings["topology"] = {
            platform.pairs[pair].coupler.name: list(pair)
            for pair in platform.ordered_pairs
        }

    settings["native_gates"] = dump_native_gates(
        platform.qubits, platform.pairs, platform.couplers
    )
    settings["characterization"] = dump_characterization(
        platform.qubits, platform.couplers
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
        if qubit.kernel is not None:
            kernels[qubit.name] = qubit.kernel

    # dump only if not None
    if kernels:
        kernels.dump(path)


def dump_platform(platform: Platform, path: Path):
    """Platform serialization as runcard (json) and kernels (npz).

    Args:
        platform (qibolab.platform.Platform): The platform to be serialized.
        path (pathlib.Path): Path where json and npz will be dumped.
    """

    dump_kernels(platform=platform, path=path)
    dump_runcard(platform=platform, path=path)
