"""Serialization utilities."""

import base64
import io
from typing import Annotated, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, PlainSerializer, PlainValidator


def ndarray_serialize(ar: npt.NDArray) -> str:
    """Serialize array to string."""
    buffer = io.BytesIO()
    np.save(buffer, ar)
    buffer.seek(0)
    return base64.standard_b64encode(buffer.read()).decode()


def ndarray_deserialize(x: Union[str, npt.NDArray]) -> npt.NDArray:
    """Deserialize array."""
    if isinstance(x, np.ndarray):
        return x

    buffer = io.BytesIO()
    buffer.write(base64.standard_b64decode(x))
    buffer.seek(0)
    return np.load(buffer)


NdArray = Annotated[
    npt.NDArray,
    PlainValidator(ndarray_deserialize),
    PlainSerializer(ndarray_serialize, return_type=str),
]
"""Pydantic-compatible array representation."""


class Model(BaseModel):
    """Global qibolab model, holding common configurations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
