import pytest

from qibolab.platform import Platform


def pytest_addoption(parser):
    parser.addoption("--platforms", type=str, action="store", default=None, help="qpu platforms to test on")
    parser.addoption("--address", type=str, action="store", default=None, help="address for the QM simulator")


def pytest_configure(config):
    config.addinivalue_line("markers", "qpu: mark tests that require qpu")


def pytest_generate_tests(metafunc):
    platforms = metafunc.config.option.platforms
    platforms = [] if platforms is None else platforms.split(",")

    if metafunc.module.__name__ == "qibolab.tests.test_instruments_rohde_schwarz":
        pytest.skip("Skipping Rohde Schwarz tests because it is not available in qpu5q.")

    address = metafunc.config.option.address
    if metafunc.module.__name__ == "qibolab.tests.test_instruments_qmsim":
        if address is None:
            pytest.skip("Skipping QM simulator tests because address was not provided.")
        else:
            metafunc.parametrize("address", [address])

    if "platform_name" in metafunc.fixturenames:
        if "qubit" in metafunc.fixturenames:
            # TODO: Do backend initialization here instead of every test (currently does not work)
            qubits = [(n, q) for n in platforms for q in range(Platform(n).nqubits)]
            metafunc.parametrize("platform_name,qubit", qubits)
        else:
            metafunc.parametrize("platform_name", platforms)
