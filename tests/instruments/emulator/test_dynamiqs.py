from pathlib import Path

import numpy as np
import pytest

import qibolab
from qibolab._core.execution_parameters import AveragingMode
from qibolab._core.instruments.emulator.engine import DynamiqsEngine, QutipEngine

pytest.importorskip("dynamiqs")

HERE = Path(__file__).parent


def test_dynamiqs_operator_arithmetic():
    engine = DynamiqsEngine()

    projector = engine.basis(2, 0) * engine.basis(2, 0).dag()
    np.testing.assert_allclose(projector.full(), np.diag([1, 0]))

    number = engine.create(3) * engine.destroy(3)
    np.testing.assert_allclose(number.full(), np.diag([0, 1, 2]), atol=1e-15)


def test_dynamiqs_expand_matches_qutip():
    dynamiqs = DynamiqsEngine()
    qutip = QutipEngine()

    dynamiqs_operator = dynamiqs.tensor([dynamiqs.destroy(2), dynamiqs.create(2)])
    qutip_operator = qutip.tensor([qutip.destroy(2), qutip.create(2)])

    np.testing.assert_allclose(
        dynamiqs.expand(dynamiqs_operator, [2, 3, 2], [0, 2]).full(),
        qutip.expand(qutip_operator, [2, 3, 2], [0, 2]).full(),
        atol=1e-15,
    )


def test_dynamiqs_engine_sequence(monkeypatch):
    monkeypatch.setenv("QIBOLAB_PLATFORMS", str(HERE / "platforms"))
    platform = qibolab.create_platform("qubit")
    platform.instruments["dummy"].engine = DynamiqsEngine()

    q0 = platform.natives.single_qubit[0]
    seq = q0.RX() | q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id

    result = platform.execute(
        [seq],
        averaging_mode=AveragingMode.CYCLIC,
    )

    assert pytest.approx(result[acq_handle], abs=5e-2) == 1
