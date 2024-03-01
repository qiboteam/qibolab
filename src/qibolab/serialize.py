"""Helper methods for loading and saving to runcards.

The format of runcards in the ``qiboteam/qibolab_platforms_qrc``
repository is assumed here. See :ref:`Using runcards <using_runcards>`
example for more details.
"""

import json
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Tuple

from qibolab.couplers import Coupler
from qibolab.kernels import Kernels
from qibolab.native import SingleQubitNatives, TwoQubitNatives
from qibolab.platform import (
    CouplerMap,
    InstrumentMap,
    Platform,
    QubitMap,
    QubitPairMap,
    Settings,
)
from qibolab.pulses import Delay, Pulse, PulseSequence, PulseType
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
    qubits = {}
    for q, char in runcard["characterization"]["single_qubit"].items():
        raw_qubit = Qubit(json.loads(q), **char)
        raw_qubit.crosstalk_matrix = {
            json.loads(key): value for key, value in raw_qubit.crosstalk_matrix.items()
        }
        qubits[json.loads(q)] = raw_qubit

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


def _load_pulse(pulse_kwargs, qubit):
    pulse_type = pulse_kwargs.pop("type")
    if "coupler" in pulse_kwargs:
        q = pulse_kwargs.pop("coupler", qubit.name)
    else:
        q = pulse_kwargs.pop("qubit", qubit.name)

    if pulse_type == "dl":
        return Delay(**pulse_kwargs)
    return Pulse(**pulse_kwargs, type=pulse_type, qubit=q)


def _load_single_qubit_natives(qubit, gates) -> SingleQubitNatives:
    """Parse native gates of the qubit from the runcard.

    Args:
        qubit (:class:`qibolab.qubits.Qubit`): Qubit object that the
            native gates are acting on.
        gates (dict): Dictionary with native gate pulse parameters as loaded
            from the runcard.
    """
    return SingleQubitNatives(
        **{name: _load_pulse(kwargs, qubit) for name, kwargs in gates.items()}
    )


def _load_two_qubit_natives(qubits, couplers, gates) -> TwoQubitNatives:
    sequences = {}
    for name, seq_kwargs in gates.items():
        if isinstance(seq_kwargs, dict):
            seq_kwargs = [seq_kwargs]

        sequence = PulseSequence()
        virtual_z_phases = defaultdict(int)
        for kwargs in seq_kwargs:
            _type = kwargs["type"]
            if _type == "virtual_z":
                q = kwargs["qubit"]
                virtual_z_phases[q] += kwargs["phase"]
            else:
                if "coupler" in kwargs:
                    qubit = couplers[kwargs["coupler"]]
                else:
                    qubit = qubits[kwargs["qubit"]]
                sequence.append(_load_pulse(kwargs, qubit))

        sequences[name] = (sequence, virtual_z_phases)
        return TwoQubitNatives(**sequences)


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
        qubit = qubits[json.loads(q)]
        qubit.native_gates = _load_single_qubit_natives(qubit, gates)

    for c, gates in native_gates.get("coupler", {}).items():
        coupler = couplers[json.loads(c)]
        coupler.native_gates = _load_single_qubit_natives(coupler, gates)

    # register two-qubit native gates to ``QubitPair`` objects
    for pair, gatedict in native_gates.get("two_qubit", {}).items():
        q0, q1 = tuple(int(q) if q.isdigit() else q for q in pair.split("-"))
        native_gates = _load_two_qubit_natives(qubits, couplers, gatedict)
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


def _dump_pulse(pulse: Pulse):
    data = asdict(pulse)
    if pulse.type in (PulseType.FLUX, PulseType.COUPLERFLUX):
        del data["frequency"]
    data["shape"] = str(pulse.shape)
    data["type"] = data["type"].value
    del data["channel"]
    del data["relative_phase"]
    return data


def _dump_single_qubit_natives(natives: SingleQubitNatives):
    data = {}
    for fld in fields(natives):
        pulse = getattr(natives, fld.name)
        if pulse is not None:
            data[fld.name] = _dump_pulse(pulse)
            del data[fld.name]["qubit"]
    return data


def _dump_two_qubit_natives(natives: TwoQubitNatives):
    data = {}
    for fld in fields(natives):
        if getattr(natives, fld.name) is None:
            continue
        sequence, virtual_z_phases = getattr(natives, fld.name)
        data[fld.name] = []
        for pulse in sequence:
            pulse_serial = _dump_pulse(pulse)
            if pulse.type == PulseType.COUPLERFLUX:
                pulse_serial["coupler"] = pulse_serial["qubit"]
            data[fld.name].append(pulse_serial)
        data[fld.name].extend(
            {"type": "virtual_z", "phase": phase, "qubit": q}
            for q, phase in virtual_z_phases.items()
        )
    return data


def dump_native_gates(
    qubits: QubitMap, pairs: QubitPairMap, couplers: CouplerMap = None
) -> dict:
    """Dump native gates section to dictionary following the runcard format,
    using qubit and pair objects."""
    # single-qubit native gates
    native_gates = {
        "single_qubit": {
            json.dumps(q): _dump_single_qubit_natives(qubit.native_gates)
            for q, qubit in qubits.items()
        }
    }

    if couplers:
        native_gates["coupler"] = {
            json.dumps(c): _dump_single_qubit_natives(coupler.native_gates)
            for c, coupler in couplers.items()
        }

    # two-qubit native gates
    native_gates["two_qubit"] = {}
    for pair in pairs.values():
        natives = _dump_two_qubit_natives(pair.native_gates)
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
