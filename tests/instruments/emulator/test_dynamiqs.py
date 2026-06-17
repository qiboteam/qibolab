"""Testing Dynamiqs emulator engine."""

from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import make_interp_spline

from qibolab import AcquisitionType, AveragingMode, create_platform
from qibolab._core.instruments.emulator.emulator import EmulatorController
from qibolab._core.instruments.emulator.engine import DynamiqsEngine, QutipEngine
from qibolab._core.instruments.emulator.engine.abstract import OperatorEvolution

pytest.importorskip("dynamiqs")

from qibolab._core.instruments.emulator.engine.abstract import _spline_function

PLATFORMS_DIR = Path(__file__).parent / "platforms"
PLATFORM_NAMES = sorted(p.name for p in PLATFORMS_DIR.iterdir() if p.is_dir())


def test_dynamiqs_operator_algebra_matches_qutip():
    dynamiqs = DynamiqsEngine()
    qutip = QutipEngine()

    assert np.allclose(dynamiqs.destroy(3).full(), qutip.destroy(3).full())
    assert np.allclose(dynamiqs.create(3).full(), qutip.create(3).full())
    assert np.allclose(dynamiqs.identity(3).full(), qutip.identity(3).full())
    assert np.allclose(dynamiqs.basis(3, 1).full(), qutip.basis(3, 1).full())
    assert np.allclose(
        (dynamiqs.create(3) * dynamiqs.destroy(3)).full(),
        (qutip.create(3) * qutip.destroy(3)).full(),
    )


def test_dynamiqs_matmul_matches_qutip():
    dynamiqs = DynamiqsEngine()
    qutip = QutipEngine()

    assert np.allclose(
        (dynamiqs.create(3) @ dynamiqs.destroy(3)).full(),
        (qutip.create(3) @ qutip.destroy(3)).full(),
    )
    assert np.allclose(
        (dynamiqs.create(3) @ dynamiqs.destroy(3)).full(),
        (dynamiqs.create(3) * dynamiqs.destroy(3)).full(),
    )


@pytest.mark.parametrize("target", [0, 1])
def test_dynamiqs_expand_matches_qutip(target):
    dynamiqs = DynamiqsEngine()
    qutip = QutipEngine()

    assert np.allclose(
        dynamiqs.expand(dynamiqs.destroy(2), [2, 2], target).full(),
        qutip.expand(qutip.destroy(2), [2, 2], target).full(),
    )


def test_dynamiqs_two_body_expand_matches_qutip():
    dynamiqs = DynamiqsEngine()
    qutip = QutipEngine()
    dynamiqs_pair = dynamiqs.tensor([dynamiqs.destroy(2), dynamiqs.create(2)])
    qutip_pair = qutip.tensor([qutip.destroy(2), qutip.create(2)])

    assert np.allclose(
        dynamiqs.expand(dynamiqs_pair, [2, 2], [0, 1]).full(),
        qutip.expand(qutip_pair, [2, 2], [0, 1]).full(),
    )


def test_spline_function_matches_scipy():
    """The JAX-traceable conversion reproduces the cubic spline exactly."""
    times = np.linspace(0.0, 20.0, 101)
    values = np.sin(np.pi * times / 20.0) ** 2 * np.cos(2 * np.pi * 0.3 * times)
    spline = make_interp_spline(times, values, k=3)
    evaluate = _spline_function(spline)

    dense = np.linspace(0.0, 20.0, 1009)
    converted = np.array([float(evaluate(t)) for t in dense])
    np.testing.assert_allclose(converted, spline(dense), atol=1e-10)


def test_dynamiqs_evolution_matches_qutip():
    dynamiqs = DynamiqsEngine()
    qutip = QutipEngine()
    times = np.linspace(0, 1, 5)

    dynamiqs_result = dynamiqs.evolve(
        hamiltonian=dynamiqs.identity(2) * 0,
        initial_state=dynamiqs.basis(2, 1),
        time=times,
        collapse_operators=[0.2 * dynamiqs.destroy(2)],
    )
    qutip_result = qutip.evolve(
        hamiltonian=qutip.identity(2) * 0,
        initial_state=qutip.basis(2, 1),
        time=times,
        collapse_operators=[0.2 * qutip.destroy(2)],
    )

    assert np.allclose(
        dynamiqs_result.states[-1].full(),
        qutip_result.states[-1].full(),
        atol=1e-4,
    )


def _driven_evolution(engine, **kwargs):
    """Driven three-level transmon: static detuning and anharmonicity, a
    cubic-spline drive with a resonant carrier, and amplitude damping."""
    a = engine.destroy(3)
    hamiltonian = 2 * np.pi * 0.2 * (a.dag() * a) + 2 * np.pi * (-0.1) * (
        a.dag() * a.dag() * a * a
    )
    times = np.linspace(0.0, 20.0, 201)
    coefficient = (
        2
        * np.pi
        * 0.0125
        * np.sin(np.pi * times / 20.0) ** 2
        * np.cos(2 * np.pi * 0.2 * times)
    )
    evolution = OperatorEvolution(
        operators=[[a + a.dag(), make_interp_spline(times, coefficient, k=3)]]
    )
    result = engine.evolve(
        hamiltonian=hamiltonian,
        initial_state=engine.basis(3, 0),
        time=np.linspace(0.0, 20.0, 5),
        time_hamiltonian=evolution,
        collapse_operators=[0.005 * a],
        **kwargs,
    )
    return np.stack([state.full() for state in result.states])


@pytest.mark.parametrize(
    "engine",
    [
        DynamiqsEngine(),
        DynamiqsEngine(method="fixed"),
        DynamiqsEngine(method="fixed", fixed_step_dt=2e-2),
    ],
    ids=["adaptive", "fixed", "fixed-coarse"],
)
def test_dynamiqs_driven_evolution_matches_qutip(engine):
    reference = _driven_evolution(QutipEngine())
    states = _driven_evolution(engine)

    np.testing.assert_allclose(states, reference, atol=1e-4)


def test_dynamiqs_dump_and_load_results(tmp_path):
    engine = DynamiqsEngine()
    states = _driven_evolution(engine, save_evolution=tmp_path)

    hamiltonians, dumped = engine.load_results(tmp_path)
    assert hamiltonians["hamiltonians"].shape == (5, 3, 3)
    np.testing.assert_allclose(dumped["times"], np.linspace(0.0, 20.0, 5))
    np.testing.assert_allclose(dumped["states"], states, atol=1e-12)

    # a second dump in the same directory gets an incremented index
    _driven_evolution(engine, save_evolution=tmp_path)
    assert (tmp_path / "State_Evolution_1.npz").is_file()


@pytest.mark.parametrize("platform_name", PLATFORM_NAMES)
def test_dynamiqs_engine_executes_platforms(monkeypatch, platform_name):
    monkeypatch.setenv("QIBOLAB_PLATFORMS", str(PLATFORMS_DIR))
    platform = create_platform(platform_name)
    for instrument in platform.instruments.values():
        if isinstance(instrument, EmulatorController):
            instrument.engine = DynamiqsEngine()

    q0 = platform.natives.single_qubit[0]
    seq = q0.RX() | q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    result = platform.execute(
        [seq],
        nshots=100,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    assert result[acq_handle].shape == (2,)
    assert result[acq_handle][0] == pytest.approx(1, abs=5e-2)
    assert result[acq_handle][1] == pytest.approx(0, abs=1e-2)
