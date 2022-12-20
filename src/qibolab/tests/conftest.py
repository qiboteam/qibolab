import pytest

from qibolab.platform import Platform


def pytest_addoption(parser):
    parser.addoption("--platforms", type=str, action="store", default=None, help="qpu platforms to test on")


def pytest_configure(config):
    config.addinivalue_line("markers", "qpu: mark tests that require qpu")


def pytest_generate_tests(metafunc):
    platforms = metafunc.config.option.platforms
    platforms = [] if platforms is None else platforms.split(",")

    if "platform_name" in metafunc.fixturenames:
        if "qubit" in metafunc.fixturenames:
            # TODO: Do backend initialization here instead of every test (currently does not work)
            qubits = [(n, q) for n in platforms for q in range(Platform(n).nqubits)]
            metafunc.parametrize("platform_name,qubit", qubits)
        else:
            metafunc.parametrize("platform_name", platforms)
