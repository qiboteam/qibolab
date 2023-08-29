"""Helper methods for loading and saving to runcards.

The format of runcards in the ``qiboteam/qibolab_platforms_qrc``
repository is assumed here. See :ref:`Using runcards <using_runcards>`
example for more details.
"""
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import yaml

from qibolab.couplers import Coupler, CouplerPair
from qibolab.native import SingleQubitNatives, TwoQubitNatives
from qibolab.platform import (
    CouplerMap,
    CouplerPairMap,
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


def load_qubits(runcard: dict) -> Tuple[QubitMap, QubitPairMap]:
    """Load qubits and pairs from the runcard.

    Uses the native gate and characterization sections of the runcard
    to parse the :class:`qibolab.qubits.Qubit` and :class:`qibolab.qubits.QubitPair`
    objects.
    """
    qubits = {q: Qubit(q, **char) for q, char in runcard["characterization"]["single_qubit"].items()}

    pairs = {}
    for pair in runcard["topology"]:
        pair = tuple(sorted(pair))
        pairs[pair] = QubitPair(qubits[pair[0]], qubits[pair[1]])

    return qubits, pairs


def load_couplers(runcard: dict) -> Tuple[CouplerMap, CouplerPairMap]:
    """Load couplers and pairs from the runcard.

    Uses the native gate and characterization sections of the runcard
    to parse the :class:`qibolab.qubits.Qubit` and :class:`qibolab.qubits.QubitPair`
    and the the :class:`qibolab.coupler.Coupler` and :class:`qibolab.coupler.CouplerPair`
    objects.
    """
    qubits = {q: Qubit(q, **char) for q, char in runcard["characterization"]["single_qubit"].items()}
    couplers = {c: Coupler(c, **char) for c, char in runcard["characterization"]["coupler"].items()}

    coupler_pairs = {}
    for pair, coupler in zip(runcard["topology"], runcard["couplers"]):
        pair = tuple(sorted(pair))
        # Fancier ordering for couplers
        coupler_pairs[pair] = CouplerPair(couplers[coupler], qubits[pair[0]], qubits[pair[1]])

    return couplers, coupler_pairs


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
    # register two-qubit native gates to ``QubitPair`` objects
    for pair, gatedict in native_gates.get("two_qubit", {}).items():
        pair = tuple(sorted(int(q) if q.isdigit() else q for q in pair.split("-")))
        pairs[pair].native_gates = TwoQubitNatives.from_dict(qubits, couplers, gatedict)

    return qubits, pairs


def dump_qubits(qubits: QubitMap, pairs: QubitPairMap) -> dict:
    """Dump qubit and pair objects to a dictionary following the runcard format."""
    native_gates = {
        "single_qubit": {q: qubit.native_gates.raw for q, qubit in qubits.items()},
        "two_qubit": {},
    }
    # add two-qubit native gates
    for p, pair in pairs.items():
        natives = pair.native_gates.raw
        if len(natives) > 0:
            native_gates["two_qubit"][f"{p[0]}-{p[1]}"] = natives
    # add qubit characterization section
    return {
        "native_gates": native_gates,
        "characterization": {"single_qubit": {q: qubit.characterization for q, qubit in qubits.items()}},
    }


# TODO: Couplers would need to be associated to qubits in the runcard
def dump_couplers(couplers: CouplerMap, coupler_pairs: CouplerPairMap) -> dict:
    """Dump coupler and coupler_pair objects to a dictionary following the runcard format."""
    # TODO:
    return {"couplers": couplers, "coupler_qubits": coupler_pairs}


def dump_runcard(platform: Platform, path: Path):
    """Serializes the platform and saves it as a yaml runcard file.

    The file saved follows the format explained in :ref:`Using runcards <using_runcards>`.

    Args:
        platform (qibolab.platform.Platform): The platform to be serialized.
        path (pathlib.Path): Path that the yaml file will be saved.
    """
    settings = {
        "nqubits": platform.nqubits,
        "qubits": list(platform.qubits),
        "settings": asdict(platform.settings),
        "topology": [list(pair) for pair in platform.pairs],
    }
    settings.update(dump_qubits(platform.qubits, platform.pairs))
    path.write_text(yaml.dump(settings, sort_keys=False, indent=4, default_flow_style=None))
