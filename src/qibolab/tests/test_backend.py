import pytest


def test_backend_init():
    from qibolab.backend import QibolabBackend
    from qibolab.platforms.qbloxplatform import QBloxPlatform
    backend = QibolabBackend()
    assert backend.is_hardware
    assert isinstance(backend.platform, QBloxPlatform)


def test_backend_platform():
    from qibolab.backend import QibolabBackend
    from qibolab.platforms.qbloxplatform import QBloxPlatform
    from qibolab.platforms.icplatform import ICPlatform
    backend = QibolabBackend()
    original_platform = backend.get_platform()
    
    # FIXME: Enable this test when qili runcard is available
    #backend.set_platform("qili")
    #assert backend.platform.name == "qili"
    #assert isinstance(backend.platform, QBloxPlatform)
    
    backend.set_platform("icarusq")
    assert backend.platform.name == "icarusq"
    assert isinstance(backend.platform, ICPlatform)

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