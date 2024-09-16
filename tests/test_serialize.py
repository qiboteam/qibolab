import numpy as np
from pydantic import BaseModel, ConfigDict

from qibolab._core.serialize import NdArray, eq


class ArrayModel(BaseModel):
    ar: NdArray

    model_config = ConfigDict(arbitrary_types_allowed=True)


def test_equality():
    assert eq(ArrayModel(ar=np.arange(10)), ArrayModel(ar=np.arange(10)))
    assert not eq(ArrayModel(ar=np.arange(10)), ArrayModel(ar=np.arange(11)))
    ar = np.arange(10)
    ar[5:] = 42
    assert not eq(ArrayModel(ar=np.arange(10)), ArrayModel(ar=ar))

    assert not eq(ArrayModel(ar=np.arange(10)), ArrayModel(ar=np.ones((10, 2))))
    assert eq(ArrayModel(ar=np.ones((10, 2))), ArrayModel(ar=np.ones((10, 2))))
