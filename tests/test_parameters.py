import pytest

from qibolab.native import FixedSequenceFactory, TwoQubitNatives
from qibolab.parameters import TwoQubitContainer


def test_two_qubit_container():
    """The container guarantees access to symmetric interactions.

    Swapped indexing is working (only with getitem, not other dict
    methods) if all the registered natives are symmetric.
    """
    symmetric = TwoQubitContainer({(0, 1): TwoQubitNatives(CZ=FixedSequenceFactory())})
    assert symmetric[1, 0].CZ is not None

    asymmetric = TwoQubitContainer(
        {(0, 1): TwoQubitNatives(CNOT=FixedSequenceFactory())}
    )
    with pytest.raises(KeyError):
        asymmetric[(1, 0)]

    empty = TwoQubitContainer({(0, 1): TwoQubitNatives()})
    assert empty[(1, 0)] is not None
