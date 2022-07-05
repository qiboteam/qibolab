import pytest
from qibolab.backend import QibolabBackend
from qibolab.platforms.multiqubit import MultiqubitPlatform
from qibolab.platforms.icplatform import ICPlatform

def test_backend_init():
    backend = QibolabBackend()
    assert backend.is_hardware
    assert isinstance(backend.platform, MultiqubitPlatform)


@pytest.mark.xfail
@pytest.mark.parametrize("platform_name", ['tiiq', 'qili', 'multiqubit', 'icarusq'])
def test_backend_platform_multiqubit(platform_name):
    backend = QibolabBackend()
    original_platform = backend.get_platform()
    
    backend.set_platform(platform_name)
    assert backend.platform.name == platform_name
    assert isinstance(backend.platform, MultiqubitPlatform)

    with pytest.raises(RuntimeError):
        backend.set_platform("nonexistent")
        
    backend.set_platform(original_platform)


def test_backend_circuit_class():
    from qibolab.backend import QibolabBackend
    from qibolab.circuit import HardwareCircuit
    backend = QibolabBackend()
    assert backend.circuit_class() == HardwareCircuit
    with pytest.raises(NotImplementedError):
        backend.circuit_class(accelerators={"/GPU:0": 2})
    with pytest.raises(NotImplementedError):
        backend.circuit_class(density_matrix=True)


# ``QibolabBackend.create_gate`` is tested via ``test_gates.py``