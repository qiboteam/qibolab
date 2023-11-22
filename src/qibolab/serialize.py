"""Helper methods for loading and saving to runcards.

The format of runcards in the ``qiboteam/qibolab_platforms_qrc``
repository is assumed here. See :ref:`Using runcards <using_runcards>`
example for more details.
"""
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import yaml

from qibolab.couplers import Coupler
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


def load_runcard(path: Path) -> dict:
    """Load runcard YAML to a dictionary."""
    return yaml.safe_load(path.read_text())


def load_settings(runcard: dict) -> Settings:
    """Load platform settings section from the runcard."""
    return Settings(**runcard["settings"])


def load_qubits(runcard: dict) -> Tuple[QubitMap, CouplerMap, QubitPairMap]:
    """Load qubits and pairs from the runcard.

    Uses the native gate and characterization sections of the runcard
    to parse the :class:`qibolab.qubits.Qubit` and :class:`qibolab.qubits.QubitPair`
    objects.
    """
    qubits = {q: Qubit(q, **char) for q, char in runcard["characterization"]["single_qubit"].items()}

    couplers = {}
    pairs = {}
    if "coupler" in runcard["characterization"]:
        couplers = {c: Coupler(c, **char) for c, char in runcard["characterization"]["coupler"].items()}

        for c, pair in runcard["topology"].items():
            pair = tuple(sorted(pair))
            pairs[pair] = QubitPair(qubits[pair[0]], qubits[pair[1]], couplers[c])
    else:
        for pair in runcard["topology"]:
            pair = tuple(sorted(pair))
            pairs[pair] = QubitPair(qubits[pair[0]], qubits[pair[1]], None)

    qubits, pairs, couplers = register_gates(runcard, qubits, pairs, couplers)

    return qubits, couplers, pairs


# This creates the compiler error
def register_gates(
    runcard: dict, qubits: QubitMap, pairs: QubitPairMap, couplers: CouplerMap = None
) -> Tuple[QubitMap, QubitPairMap]:
    """Register single qubit native gates to ``Qubit`` objects from the runcard.

    Uses the native gate and characterization sections of the runcard
    to parse the :class:`qibolab.qubits.Qubit` and :class:`qibolab.qubits.QubitPair`
    objects.
    """

    native_gates = runcard.get("native_gates", {})
    for q, gates in native_gates.get("single_qubit", {}).items():
        qubits[q].native_gates = SingleQubitNatives.from_dict(qubits[q], gates)
    for c, gates in native_gates.get("coupler", {}).items():
        couplers[c].native_pulse = CouplerNatives.from_dict(couplers[c], gates)
    # register two-qubit native gates to ``QubitPair`` objects
    for pair, gatedict in native_gates.get("two_qubit", {}).items():
        pair = tuple(sorted(int(q) if q.isdigit() else q for q in pair.split("-")))
        pairs[pair].native_gates = TwoQubitNatives.from_dict(qubits, couplers, gatedict)

    return qubits, pairs, couplers


def load_instrument_settings(runcard: dict, instruments: InstrumentMap) -> InstrumentMap:
    """Setup instruments according to the settings given in the runcard."""
    for name, settings in runcard.get("instruments", {}).items():
        instruments[name].setup(**settings)
    return instruments


def dump_qubits(qubits: QubitMap, pairs: QubitPairMap, couplers: CouplerMap = None) -> dict:
    """Dump qubit and pair objects to a dictionary following the runcard format."""

    native_gates = {"single_qubit": {q: qubit.native_gates.raw for q, qubit in qubits.items()}}
    if couplers:
        native_gates["coupler"] = {c: coupler.native_pulse.raw for c, coupler in couplers.items()}
    native_gates["two_qubit"] = {}

    # add two-qubit native gates
    for p, pair in pairs.items():
        natives = pair.native_gates.raw
        if len(natives) > 0:
            native_gates["two_qubit"][f"{p[0]}-{p[1]}"] = natives
    # add qubit characterization section
    characterization = {
        "single_qubit": {q: qubit.characterization for q, qubit in qubits.items()},
    }
    if couplers:
        characterization["coupler"] = {c.name: {"sweetspot": c.sweetspot} for c in couplers.values()}

    return {
        "native_gates": native_gates,
        "characterization": characterization,
    }


def dump_instruments(instruments: InstrumentMap) -> dict:
    """Dump instrument settings to a dictionary following the runcard format."""
    return {
        name: asdict(instrument.settings, dict_factory=instrument.settings.dict_factory)
        for name, instrument in instruments.items()
        if instrument.settings is not None
    }


def dump_runcard(platform: Platform, path: Path):
    """Serializes the platform and saves it as a yaml runcard file.

    The file saved follows the format explained in :ref:`Using runcards <using_runcards>`.

    Args:
        platform (qibolab.platform.Platform): The platform to be serialized.
        path (pathlib.Path): Path that the yaml file will be saved.
    """

    settings = {
        "nqubits": platform.nqubits,
        "settings": asdict(platform.settings),
        "qubits": list(platform.qubits),
        "topology": [list(pair) for pair in platform.pairs],
        "instruments": dump_instruments(platform.instruments),
    }
    if platform.couplers:
        settings["couplers"] = list(platform.couplers)
        settings["topology"] = {coupler: list(pair) for pair, coupler in zip(platform.pairs, platform.couplers)}

    settings.update(dump_qubits(platform.qubits, platform.pairs, platform.couplers))
    path.write_text(yaml.dump(settings, sort_keys=False, indent=4, default_flow_style=None))
