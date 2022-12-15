import pytest

from qibolab.platform import Platform


def pytest_addoption(parser):
    parser.addoption("--platforms", type=str, action="store", default=None, help="qpu platforms to test on")
    parser.addoption("--qmsim", type=str, action="store", default=None, help="cloud address for QM simulator")


def pytest_configure(config):
    config.addinivalue_line("markers", "qpu: mark tests that require qpu")


def pytest_generate_tests(metafunc):
    platforms = metafunc.config.option.platforms
    platforms = [] if platforms is None else platforms.split(",")
    qmsim = metafunc.config.option.qmsim
    qmsim = [] if qmsim is None else [qmsim]

    # TODO: Enable tests for R&S as it is used for Quantum Machines
    if metafunc.module.__name__ == "qibolab.tests.test_instruments_rohde_schwarz":
        pytest.skip("Skipping Rohde Schwarz tests because it is not available in qpu5q.")

    if "qmsim_address" in metafunc.fixturenames:
        metafunc.parametrize("qmsim_address", qmsim)

    if "platform_name" in metafunc.fixturenames:
        if "qubit" in metafunc.fixturenames:
            # TODO: Do backend initialization here instead of every test (currently does not work)
            qubits = [(n, q) for n in platforms for q in range(Platform(n).nqubits)]
            metafunc.parametrize("platform_name,qubit", qubits)
        else:
            metafunc.parametrize("platform_name", platforms)
