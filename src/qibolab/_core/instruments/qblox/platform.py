from collections import defaultdict
from typing import Optional, Union

from qibolab._core.components.channels import (
    AcquisitionChannel,
    Channel,
    DcChannel,
    IqChannel,
)

__all__ = ["map_ports"]


def _eltype(el: str):
    return "qubits" if el[0] == "q" else "couplers"


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
            "acquisition": AcquisitionChannel(path=f"{slot}/o{port[2:]}"),
        }
    port = f"o{port}" if isinstance(port, int) else port
    name, cls = _chtype(mod, port[0] == "i")
    return {name: cls(path=f"{slot}/{port}")}


def _premap(cluster: dict):
    d = {
        "qubits": defaultdict(lambda: defaultdict(dict)),
        "couplers": defaultdict(lambda: defaultdict(dict)),
    }

    for mod, props in cluster.items():
        slot = props[0]
        for port, els in props[1].items():
            for el in els:
                nel = el[1:]
                d[_eltype(el)][nel] |= _port_channels(mod, port, slot)

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
    for kind, elements in [("qubits", qubits), ("couplers", couplers)]:
        for name, el in elements.items():
            for chname, ch in premap[kind][name].items():
                channels[getattr(el, chname)] = ch
            if kind == "qubits":
                channels[el.acquisition] = channels[el.acquisition].model_copy(
                    update={"probe": el.probe}
                )
    return channels
