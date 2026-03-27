from typing import Annotated, TypeAlias, Union

import numpy as np
import numpy.typing as npt
from pydantic import BeforeValidator, Field, PlainSerializer

__all__ = ["Result"]

QubitId = Annotated[Union[int, str], Field(union_mode="left_to_right")]
"""Qubit name."""

QubitPairKey: TypeAlias = tuple[QubitId, QubitId]
"""Two-qubit pair key for static typing."""


ChannelId = str
"""Unique identifier for a channel."""


StateId = int
"""State identifier."""


def _split(pair: Union[str, tuple]) -> tuple[str, str]:
    tupled_pair = pair if isinstance(pair, tuple) else tuple(pair.split("-"))
    assert len(tupled_pair) == 2
    return tupled_pair


def _join(pair: tuple[str, str]) -> str:
    return f"{pair[0]}-{pair[1]}"


TransitionId = Annotated[
    tuple[StateId, StateId], BeforeValidator(_split), PlainSerializer(_join)
]
"""Identifier for a state transition."""


QubitPairId = Annotated[QubitPairKey, BeforeValidator(_split), PlainSerializer(_join)]
"""Two-qubit active interaction identifier."""


Result = npt.NDArray[np.float64]
"""An array of results returned by instruments."""
