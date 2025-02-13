import numpy as np
from pydantic import BaseModel, ConfigDict

from qibolab._core.serialize import ArrayList, NdArray, eq


class ArrayModel(BaseModel):
    ar: NdArray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TestNdArray:
    def test_equality(self):
        assert eq(ArrayModel(ar=np.arange(10)), ArrayModel(ar=np.arange(10)))
        assert not eq(ArrayModel(ar=np.arange(10)), ArrayModel(ar=np.arange(11)))
        ar = np.arange(10)
        ar[5:] = 42
        assert not eq(ArrayModel(ar=np.arange(10)), ArrayModel(ar=ar))

        assert not eq(ArrayModel(ar=np.arange(10)), ArrayModel(ar=np.ones((10, 2))))
        assert eq(ArrayModel(ar=np.ones((10, 2))), ArrayModel(ar=np.ones((10, 2))))


class ArrayListModel(BaseModel):
    ar: ArrayList

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TestArrayList:
    def test_equality(self):
        assert eq(ArrayListModel(ar=np.arange(10)), ArrayListModel(ar=np.arange(10)))
        assert not eq(
            ArrayListModel(ar=np.arange(10)), ArrayListModel(ar=np.arange(11))
        )
        ar = np.arange(10)
        ar[5:] = 42
        assert not eq(ArrayListModel(ar=np.arange(10)), ArrayListModel(ar=ar))

        assert not eq(
            ArrayListModel(ar=np.arange(10)), ArrayListModel(ar=np.ones((10, 2)))
        )
        assert eq(
            ArrayListModel(ar=np.ones((10, 2))), ArrayListModel(ar=np.ones((10, 2)))
        )

    def test_serde(self):
        lst = ArrayListModel(ar=np.arange(10)).model_dump()["ar"]
        assert isinstance(lst, list)
        assert isinstance(lst[0], int)
        assert len(lst) == 10
