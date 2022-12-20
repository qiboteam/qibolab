import pytest

from qibolab.platform import Platform


def pytest_addoption(parser):
    parser.addoption("--platform", type=str, action="store", default=None, help="qpu platform to test")


def pytest_configure(config):
    config.addinivalue_line("markers", "qpu: mark tests that require qpu")


def pytest_generate_tests(metafunc):
    platform_name = metafunc.config.option.platform

    if "platform_name" in metafunc.fixturenames:
        if "qubit" in metafunc.fixturenames:
            # TODO: Do backend initialization here instead of every test (currently does not work)
            qubits = [(platform_name, q) for q in range(Platform(platform_name).nqubits)]
            metafunc.parametrize("platform_name,qubit", qubits)
        else:
            metafunc.parametrize("platform_name", [platform_name])
