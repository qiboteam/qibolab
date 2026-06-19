"""Testing emulator GPU configuration."""

import importlib.util

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


def test_qutip_gpu_unknown_dtype():
    with pytest.raises(ValueError, match="gpu_dtype"):
        QutipEngine(device="gpu", gpu_dtype="unknown").destroy(2)


@pytest.mark.parametrize("gpu_dtype,plugin", [("jax", "qutip_jax")])
def test_qutip_gpu_requires_plugin(gpu_dtype, plugin):
    if importlib.util.find_spec(plugin) is not None:
        pytest.skip(f"{plugin} is installed in this environment.")

    with pytest.raises(ImportError, match=plugin.replace("_", "-")):
        QutipEngine(device="gpu", gpu_dtype=gpu_dtype).destroy(2)


@pytest.mark.parametrize("gpu_dtype", ["jax", "cupyd"])
def test_qutip_gpu_data_layer(gpu_dtype):
    plugin = "qutip_jax" if gpu_dtype == "jax" else "qutip_cupy"
    pytest.importorskip(plugin)
    if gpu_dtype == "jax" and not _has_jax_gpu():
        pytest.skip("CUDA-enabled JAX backend is not available.")

    engine = QutipEngine(device="gpu", gpu_dtype=gpu_dtype)
    operator = engine.create(3) * engine.destroy(3)
    assert type(operator.data).__name__ != "Dense"
    assert np.allclose(operator.full(), np.diag([0.0, 1.0, 2.0]))


def test_dynamiqs_cpu_device_configuration():
    engine = DynamiqsEngine(device="cpu", device_index=0, matmul_precision="highest")

    assert engine.device == "cpu"
    assert np.allclose(engine.destroy(3).full(), DynamiqsEngine().destroy(3).full())
