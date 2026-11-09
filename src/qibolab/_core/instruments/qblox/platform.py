import re
from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import NamedTuple

from qibolab._core.components.channels import (
    AcquisitionChannel,
    Channel,
    DcChannel,
    IqChannel,
)
from qibolab._core.identifier import ChannelId, QubitId
from qibolab._core.qubits import QubitMap
from qibolab._core.serialize import Model

__all__ = [
    "Channels",
    "Twpa",
    "Twpas",
    "build_channels",
    "infer_los",
    "infer_mixers",
    "map_ports",
]

Channels = dict[ChannelId, Channel]
Twpas = set[ChannelId]


class Twpa(Model):
    """TWPA pump, to be placed on its port in the compact representation.

    All the attributes have defaults, such that ``Twpa()`` is enough for most
    platforms, cf. :func:`build_channels`.
    """

    name: ChannelId = "twpa"
    """Identifier of the pump channel, and its configuration.

    It has to be unique: specify it explicitly when pumping multiple TWPAs.
    """
    lo: str | None = None
    """LO name, inferred as for the other output channels if not given."""
    mixer: str | None = None
    """Mixer name, inferred as for the other output channels if not given."""


Port = str | int
SlotMap = dict[Port, list[QubitId | Twpa]]
ClusterMap = dict[str, tuple[int, SlotMap]]


def _elements(cluster: ClusterMap) -> Iterator[tuple[str, int, Port, QubitId | Twpa]]:
    """Flatten the compact representation.

    Yield each element, together with its module name, slot, and port.
    """
    for mod, (slot, ports) in cluster.items():
        for port, elements in ports.items():
            for el in elements:
                yield mod, slot, port, el


def _chtype(mod: str, input: bool) -> tuple[str, type[Channel]]:
    if mod.startswith("qcm_rf"):
        return "drive", IqChannel
    if mod.startswith("qcm"):
        return "flux", DcChannel
    if mod.startswith("qrm_rf"):
        if input:
            return "acquisition", AcquisitionChannel
        return "probe", IqChannel
    raise ValueError


def _port_path(port: Port, slot: int) -> str:
    port = f"o{port}" if isinstance(port, int) else port
    return f"{slot}/{port}"


def _port_channels(mod: str, port: Port, slot: int) -> dict[str, Channel]:
    if isinstance(port, str) and port.startswith("io"):
        return {
            "probe": IqChannel(path=f"{slot}/o{port[2:]}"),
            "acquisition": AcquisitionChannel(path=f"{slot}/i{port[2:]}"),
        }
    path = _port_path(port, slot)
    name, cls = _chtype(mod, isinstance(port, str) and port[0] == "i")
    return {name: cls(path=path)}


def _premap(
    cluster: ClusterMap,
) -> tuple[dict[QubitId, dict[str, Channel]], Channels]:
    channels = defaultdict(lambda: defaultdict(dict))
    twpas: Channels = {}

    for mod, slot, port, el in _elements(cluster):
        if isinstance(el, Twpa):
            if el.name in twpas:
                raise ValueError(
                    f"Duplicate TWPA name '{el.name}': pass distinct names to "
                    "`Twpa(name=...)` when pumping multiple TWPAs."
                )
            twpas[el.name] = IqChannel(
                path=_port_path(port, slot), lo=el.lo, mixer=el.mixer
            )
        else:
            channels[el] |= _port_channels(mod, port, slot)

    return channels, twpas


def map_ports(
    cluster: ClusterMap, qubits: dict, couplers: dict | None = None
) -> tuple[Channels, Twpas]:
    """Extract channels and TWPAs from compact representation.

    Conventions:
    - each item is a module
    - the first element of each value is the module's slot ID
    - the second element is a map from ports to qubits or :class:`Twpa` instances
        - ports
            - they are `i{n}` or `o{n}` for the inputs and outputs respectively
            - `io{n}` is also allowed, to signal that both are connected (cater for the specific
              case of the QRM_RF where there are only one port of each type)
            - if it's just an integer, it is intended to be an output (equivalent to `o{n}`)
        - values
            - list of element names or :class:`Twpa` instances
            - element names are `q{name}` or `c{name}` for qubits and couplers respectively
            - multiple elements are allowed, for multiplexed ports
            - :class:`Twpa` elements are returned among the channels, and their
              identifiers in :data:`Twpas`

    .. note::

        integer qubit names are not allowed

    .. todo::

        Currently channel types are inferred from the module type, encoded in its name. At
        least an override should be allowed (per module, per port, per qubit).
    """
    if couplers is None:
        couplers = {}

    premap, twpas = _premap(cluster)

    channels: Channels = dict(twpas)
    for name, el in (qubits | couplers).items():
        for chname, ch in premap[name].items():
            channels[getattr(el, chname)] = ch
        try:
            channels[el.acquisition] = channels[el.acquisition].model_copy(
                update={"probe": el.probe}
            )
        except KeyError:
            pass
    return channels, set(twpas)


_PORT = re.compile(r"i?o?(.*)")


def _qubit_id(string: str) -> QubitId:
    try:
        return int(string)
    except ValueError:
        return string


def _digits(string: str) -> QubitId:
    res = re.search(_PORT, string)
    return _qubit_id(res[1]) if res is not None else ""


def _out_port(port: Port) -> QubitId:
    return port if isinstance(port, int) else _digits(port)


def _output_reference(mod: str, port: Port, suffix: str) -> str:
    """Name of a component of an output, inferred from its module and port."""
    return f"{mod}/o{_out_port(port)}/{suffix}"


def _infer_outputs(cluster: ClusterMap, suffix: str) -> dict[tuple[QubitId, bool], str]:
    """``cluster`` should be a mapping compatible with the same input of :func:`map_ports`.

    The result is a mapping from ``(qubit, channel)``, where ``qubit`` is the identifier,
    and ``channel`` is a boolean toggle: ``True`` for probe channels, ``False`` for
    drive.
    """
    return {
        (el, "qrm" in mod): _output_reference(mod, port, suffix)
        for mod, _, port, el in _elements(cluster)
        if "_rf" in mod and not isinstance(el, Twpa)
    }


def infer_los(cluster: ClusterMap) -> dict[tuple[QubitId, bool], str]:
    """Infer LOs names for output channels."""
    return _infer_outputs(cluster, "lo")


def infer_mixers(cluster: ClusterMap) -> dict[tuple[QubitId, bool], str]:
    """Infer mixer names for output channels."""
    return _infer_outputs(cluster, "mixer")


def _twpa_references(cluster: ClusterMap) -> dict[ChannelId, dict[str, str]]:
    """LOs and mixers of the TWPA pumps on RF modules.

    Inferred as for the other output channels, cf. :func:`_infer_outputs`, unless
    explicitly specified in their :class:`Twpa` instance.
    """
    return {
        el.name: {
            "lo": el.lo or _output_reference(mod, port, "lo"),
            "mixer": el.mixer or _output_reference(mod, port, "mixer"),
        }
        for mod, _, port, el in _elements(cluster)
        if "_rf" in mod and isinstance(el, Twpa)
    }


class _Outputs(NamedTuple):
    """LOs and mixers of the output channels, inferred from the compact representation.

    Both are mappings from ``(element, probe)``, cf. :func:`_infer_outputs`.
    """

    los: dict[tuple[QubitId, bool], str]
    mixers: dict[tuple[QubitId, bool], str]

    @classmethod
    def infer(cls, cluster: ClusterMap) -> "_Outputs":
        return cls(los=infer_los(cluster), mixers=infer_mixers(cluster))

    def references(self, element: QubitId, probe: bool) -> dict[str, str]:
        """Channel references to the LO and mixer, if any."""
        key = (element, probe)
        return {
            name: ref
            for name, ref in (
                ("lo", self.los.get(key)),
                ("mixer", self.mixers.get(key)),
            )
            if ref is not None
        }


def build_channels(
    cluster: ClusterMap,
    qubits: QubitMap,
    couplers: QubitMap | None = None,
    twpa_pumps: Mapping[QubitId, ChannelId] | None = None,
) -> tuple[Channels, Twpas]:
    """Build the complete channels from the compact representation.

    On top of :func:`map_ports`, the probe and drive channels of RF modules are connected
    to their LOs and mixers, whose names are inferred by :func:`infer_los` and
    :func:`infer_mixers`. The same applies to the TWPA pumps, unless their LO or mixer is
    explicitly specified in their :class:`Twpa` instance. Moreover, the acquisition channels are connected to the TWPA
    pumping their transmission line, as specified by ``twpa_pumps``, mapping elements
    to the identifiers of :class:`Twpa` instances in ``cluster``.
    """
    channels, twpas = map_ports(cluster, qubits, couplers)
    outputs = _Outputs.infer(cluster)
    pumps = {} if twpa_pumps is None else twpa_pumps
    unknown = {pump for pump in pumps.values() if pump not in twpas}
    if len(unknown) > 0:
        raise ValueError(f"TWPA pumps not defined in the cluster: {unknown}")

    for name, el in (dict(qubits) | dict(couplers or {})).items():
        updates: dict[ChannelId | None, dict[str, str]] = {
            el.probe: outputs.references(name, True),
            el.drive: outputs.references(name, False),
            el.acquisition: {"twpa_pump": pumps[name]} if name in pumps else {},
        }
        for id_, update in updates.items():
            if id_ is not None and id_ in channels and len(update) > 0:
                channels[id_] = channels[id_].model_copy(update=update)

    for id_, update in _twpa_references(cluster).items():
        channels[id_] = channels[id_].model_copy(update=update)

    return channels, twpas
