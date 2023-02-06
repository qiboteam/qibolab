from qibolab.platform import Platform


def pytest_addoption(parser):
    parser.addoption("--platforms", type=str, action="store", default=None, help="qpu platforms to test")


def pytest_configure(config):
    config.addinivalue_line("markers", "qpu: mark tests that require qpu")


def pytest_generate_tests(metafunc):
    platforms = metafunc.config.option.platforms
    platforms = [] if platforms is None else platforms.split(",")

    if "platform_name" in metafunc.fixturenames:
        if "qubit" in metafunc.fixturenames:
            # TODO: Do backend initialization here instead of every test (currently does not work)
            qubits = [(platform_name, q) for platform_name in platforms for q in Platform(platform_name).qubits]
            metafunc.parametrize("platform_name,qubit", qubits)
        else:
            metafunc.parametrize("platform_name", platforms)
