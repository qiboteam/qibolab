import re
from collections import defaultdict
from typing import Optional, Union

from qibolab._core.components.channels import (
    AcquisitionChannel,
    Channel,
    DcChannel,
    IqChannel,
)
from qibolab._core.identifier import QubitId

__all__ = ["infer_los", "map_ports"]


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


def _port_channels(mod: str, port: Union[int, str], slot: int) -> dict:
    if isinstance(port, str) and port.startswith("io"):
        return {
            "probe": IqChannel(path=f"{slot}/o{port[2:]}"),
            "acquisition": AcquisitionChannel(path=f"{slot}/i{port[2:]}"),
        }
    port = f"o{port}" if isinstance(port, int) else port
    name, cls = _chtype(mod, port[0] == "i")
    return {name: cls(path=f"{slot}/{port}")}


def _premap(cluster: dict):
    d = defaultdict(lambda: defaultdict(dict))

    for mod, props in cluster.items():
        slot = props[0]
        for port, els in props[1].items():
            for el in els:
                d[el] |= _port_channels(mod, port, slot)

    return d


def map_ports(cluster: dict, qubits: dict, couplers: Optional[dict] = None) -> dict:
    """Extract channels from compact representation.

    Conventions:
    - each item is a module
    - the first element of each value is the module's slot ID
    - the second element is a map from ports to qubits
        - ports
            - they are `i{n}` or `o{n}` for the inputs and outputs respectively
            - `io{n}` is also allowed, to signal that both are connected (cater for the specific
              case of the QRM_RF where there are only one port of each type)
            - if it's just an integer, it is intended to be an output (equivalent to `o{n}`)
        - values
            - list of element names
            - they are `q{name}` or `c{name}` for qubits and couplers respectively
            - multiple elements are allowed, for multiplexed ports

    .. note::

        integer qubit names are not allowed

    .. todo::

        Currently channel types are inferred from the module type, encoded in its name. At
        least an override should be allowed (per module, per port, per qubit).
    """
    if couplers is None:
        couplers = {}

    premap = _premap(cluster)

    channels = {}
    for name, el in (qubits | couplers).items():
        for chname, ch in premap[name].items():
            channels[getattr(el, chname)] = ch
        try:
            channels[el.acquisition] = channels[el.acquisition].model_copy(
                update={"probe": el.probe}
            )
        except KeyError:
            pass
    return channels


_PORT = re.compile(r"i?o?(.*)")


def _qubit_id(string: str) -> QubitId:
    try:
        return int(string)
    except ValueError:
        return string


def _digits(string: str) -> QubitId:
    res = re.search(_PORT, string)
    return _qubit_id(res[1]) if res is not None else ""


def _out_port(port: Union[str, int]) -> QubitId:
    return port if isinstance(port, int) else _digits(port)


def infer_los(cluster: dict) -> dict[tuple[QubitId, bool], str]:
    """Infer LOs names for output channels.

    ``cluster`` should be a mapping compatible with the same input of :func:`map_ports`.

    The result is a mapping from ``(qubit, channel)``, where ``qubit`` is the identifier,
    and ``channel`` is a boolean toggle: ``True`` for probe channels, ``False`` for
    drive.
    """
    return {
        (q, "qrm" in mod): f"{mod}/o{_out_port(port)}/lo"
        for mod, specs in cluster.items()
        if "_rf" in mod
        for port, qs in specs[1].items()
        for q in qs
    }
