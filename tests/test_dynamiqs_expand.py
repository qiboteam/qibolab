import numpy as np
import pytest

from qibolab._core.instruments.emulator.engine.qutip import QutipEngine
from qibolab._core.instruments.emulator.engine.dynamiqs import DynamiqsEngine


def test_op_dims_structure():
    dq = DynamiqsEngine()
    n = 2
    op = dq.tensor([dq.destroy(n), dq.create(n)])
    print("op.dims =", op.dims)
    assert tuple(op.dims) == (n, n), f"op.dims is {op.dims}, expected (2, 2)"


def test_expand_single_target_matches_qutip():
    qt, dq = QutipEngine(), DynamiqsEngine()
    n = 2
    dims = [n, n, n]
    for t in (0, 1, 2):
        e_qt = qt.expand(qt.destroy(n), dims, t).full()
        e_dq = np.asarray(dq.expand(dq.destroy(n), dims, t).to_jax())
        assert np.allclose(e_qt, e_dq), f"single-target mismatch t={t}"


def test_expand_multi_target_matches_qutip():
    qt, dq = QutipEngine(), DynamiqsEngine()
    n = 2
    dims = [n, n, n]
    op_qt = qt.tensor([qt.destroy(n), qt.create(n)]) + qt.tensor([qt.create(n), qt.destroy(n)])
    op_dq = dq.tensor([dq.destroy(n), dq.create(n)]) + dq.tensor([dq.create(n), dq.destroy(n)])
    for targets in ([0, 1], [1, 2], [0, 2], [2, 0]):
        e_qt = qt.expand(op_qt, dims, targets).full()
        e_dq = np.asarray(dq.expand(op_dq, dims, targets).to_jax())
        assert np.allclose(e_qt, e_dq), f"multi-target mismatch targets={targets}"