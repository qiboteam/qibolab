"""Testing emulator GPU configuration."""

import numpy as np
import pytest

from qibolab._core.instruments.emulator.engine import DynamiqsEngine, QutipEngine


def _has_jax_gpu() -> bool:
    import jax

    return any(device.platform == "gpu" for device in jax.devices())


def test_qutip_cpu_device_is_default():
    engine = QutipEngine()

    assert engine.device == "cpu"
    assert np.allclose(engine.destroy(3).full(), QutipEngine().destroy(3).full())


@pytest.mark.parametrize("gpu_dtype", ["jax"])
def test_qutip_gpu_data_layer(gpu_dtype):
    pytest.importorskip(gpu_dtype)
    if gpu_dtype == "jax" and not _has_jax_gpu():
        pytest.skip("CUDA-enabled JAX backend is not available.")

    engine = QutipEngine(device="gpu")
    operator = engine.create(3) * engine.destroy(3)
    assert type(operator.data).__name__ != "Dense"
    assert np.allclose(operator.full(), np.diag([0.0, 1.0, 2.0]))


def test_dynamiqs_cpu_device_configuration():
    engine = DynamiqsEngine(device="cpu", device_index=0, matmul_precision="highest")

    assert engine.device == "cpu"
    assert np.allclose(engine.destroy(3).full(), DynamiqsEngine().destroy(3).full())
