import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from qibolab._core.serialize import ArrayList, NdArray, eq


class ArrayContainer(BaseModel):
    ar: NdArray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TestNdArray:
    def test_equality(self):
        assert eq(ArrayContainer(ar=np.arange(10)), ArrayContainer(ar=np.arange(10)))
        assert not eq(
            ArrayContainer(ar=np.arange(10)), ArrayContainer(ar=np.arange(11))
        )
        ar = np.arange(10)
        ar[5:] = 42
        assert not eq(ArrayContainer(ar=np.arange(10)), ArrayContainer(ar=ar))

        assert not eq(
            ArrayContainer(ar=np.arange(10)), ArrayContainer(ar=np.ones((10, 2)))
        )
        assert eq(
            ArrayContainer(ar=np.ones((10, 2))), ArrayContainer(ar=np.ones((10, 2)))
        )


class ArrayListContainer(BaseModel):
    ar: ArrayList

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TestArrayList:
    def test_serde(self):
        lst = ArrayListContainer(ar=np.arange(10)).model_dump()["ar"]
        assert isinstance(lst, list)
        assert isinstance(lst[0], int)
        assert len(lst) == 10

    def test_equality(self):
        assert eq(
            ArrayListContainer(ar=np.arange(10)), ArrayListContainer(ar=np.arange(10))
        )
        assert not eq(
            ArrayListContainer(ar=np.arange(10)), ArrayListContainer(ar=np.arange(11))
        )
        ar = np.arange(10)
        ar[5:] = 42
        assert not eq(ArrayListContainer(ar=np.arange(10)), ArrayListContainer(ar=ar))

        assert not eq(
            ArrayListContainer(ar=np.arange(10)),
            ArrayListContainer(ar=np.ones((10, 2))),
        )
        assert eq(
            ArrayListContainer(ar=np.ones((10, 2))),
            ArrayListContainer(ar=np.ones((10, 2))),
        )


class Nested(BaseModel):
    container0: ArrayListContainer
    container1: ArrayContainer


def test_array_nested_comparison():
    n0 = Nested(
        container0=ArrayListContainer(ar=np.arange(6)),
        container1=ArrayContainer(ar=np.arange(-10, -5)),
    )
    n1 = Nested(
        container0=ArrayListContainer(ar=np.arange(7)),
        container1=ArrayContainer(ar=np.arange(-30, -27)),
    )

    # if identical, just checking the id
    assert n0 == n0
    # otherwise, it is going for the nested comparison, and failing
    with pytest.raises(ValueError):
        assert n0 != n1
    # even with an exact copy
    with pytest.raises(ValueError):
        assert n0 == n0.model_copy(deep=True)
    # but it is actually checking array identity, not just the overall object, so it is
    # even succeeding with a shallow copy
    assert n0 == n0.model_copy()

    # but not with the nested comparison
    assert not eq(n0, n1)
    # of course, not even with the deep copy
    assert eq(n0, n0.model_copy(deep=True))
