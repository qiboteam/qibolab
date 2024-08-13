"""Serialization utilities."""

import base64
import io
from typing import Annotated, TypeVar, Union

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


def eq(obj1: BaseModel, obj2: BaseModel) -> bool:
    """Compare two models with non-default equality.

    Currently, defines custom equality for NumPy arrays.
    """
    obj2d = obj2.model_dump()
    comparisons = []
    for field, value1 in obj1.model_dump().items():
        value2 = obj2d[field]
        if isinstance(value1, np.ndarray):
            comparisons.append(
                (value1.shape == value2.shape) and (value1 == value2).all()
            )

        comparisons.append(value1 == value2)

    return all(comparisons)


class Model(BaseModel):
    """Global qibolab model, holding common configurations."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


M = TypeVar("M", bound=BaseModel)


def replace(model: M, **update) -> M:
    """Replace interface for pydantic models.

    To have the same familiar syntax of :func:`dataclasses.replace`.
    """
    return model.model_copy(update=update)
