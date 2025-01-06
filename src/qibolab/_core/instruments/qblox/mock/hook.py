from typing import Any, Type

import qblox_instruments as qblox
import qcodes.instrument
from qcodes.instrument import Instrument

from .cluster import MockCluster

__all__ = ["install"]

qcodes_find_or_create_instrument = qcodes.instrument.find_or_create_instrument


def find_or_create_instrument(
    instrument_class: Type[Instrument], *args, **kwargs
) -> Any:
    if issubclass(instrument_class, qblox.Cluster):
        return MockCluster(*args, **kwargs)

    return qcodes_find_or_create_instrument(instrument_class, *args, **kwargs)


def install():
    """Replace qblox instruments instantiation with mocked elements."""
    qcodes.instrument.find_or_create_instrument = find_or_create_instrument
