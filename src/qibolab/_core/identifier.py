from typing import Annotated, Union

import numpy as np
import numpy.typing as npt
from pydantic import BeforeValidator, Field, PlainSerializer

__all__ = ["Result"]

QubitId = Annotated[Union[int, str], Field(union_mode="left_to_right")]
"""Qubit name."""

ChannelId = str
"""Unique identifier for a channel."""


StateId = int
"""State identifier."""


def _join(pair: tuple[str, str]) -> str:
    """ "Join a tuple of qubits into a string identifier. The reason for doing this is
    that keys in JSON have to be strings."""
    return f"{pair[0]}-{pair[1]}"


def _split(pair: Union[str, tuple]) -> tuple[str, str]:
    """ "This function reverts the _join operation in case TransitionId or QubitPairId
    is loaded from a JSON file where the key is the string constructed in _join.

    .. note::

        Since it is used as a BeforeValidator, the type of pair may be neither str nor
        tuple. In that case, the function will pass on pair without modification, and
        the validation will fail later on when the type is checked against
        tuple[StateId, StateId] for TransitionId or tuple[QubitId, QubitId] for
        QubitPairId.
    """
    if isinstance(pair, str):
        a, b = pair.split("-")
        return a, b
    return pair


TransitionId = Annotated[
    tuple[StateId, StateId], BeforeValidator(_split), PlainSerializer(_join)
]
"""Identifier for a state transition."""


QubitPairId = Annotated[
    tuple[QubitId, QubitId], BeforeValidator(_split), PlainSerializer(_join)
]
"""Two-qubit active interaction identifier."""


Result = npt.NDArray[np.float64]
"""An array of results returned by instruments."""
