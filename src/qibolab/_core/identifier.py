from typing import Annotated, Any, Union

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
    """Serialize a pair identifier to a JSON-friendly key.

    Pydantic applies this serializer when dumping mappings that use
    ``TransitionId`` or ``QubitPairId`` as keys, because JSON object keys must
    be strings.
    """
    return f"{pair[0]}-{pair[1]}"


def _split(pair: Any) -> tuple[str, str]:
    """Deserialize a pair identifier previously produced by :func:`_join`.

    If ``pair`` is a string in the form ``"a-b"``, it is converted to
    ``("a", "b")`` before normal type validation. If ``pair`` is already a
    tuple, it is returned unchanged.

    As a ``BeforeValidator``, this function may also receive values of unrelated
    types (for example while validating union branches). These values are passed
    through unchanged so later validation can decide whether they are valid for
    the target type.
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
