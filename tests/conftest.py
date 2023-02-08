from qibolab.platform import Platform


def pytest_addoption(parser):
    parser.addoption(
        "--platforms",
        type=str,
        action="store",
        default="qili1q_os2,qw5q_gold",
        help="qpu platforms to test on",
    )


def pytest_generate_tests(metafunc):
    platforms = metafunc.config.option.platforms
    platforms = [] if platforms is None else platforms.split(",")

    # if metafunc.module.__name__ == "tests.test_instruments_rohde_schwarz":
    #     pytest.skip("Skipping Rohde Schwarz tests because it is not available in qpu5q.")

    if "platform_name" in metafunc.fixturenames:
        if "qubit" in metafunc.fixturenames:
            # TODO: Do backend initialization here instead of every test (currently does not work)
            qubits = [(platform_name, q) for platform_name in platforms for q in Platform(platform_name).qubits]
            metafunc.parametrize("platform_name,qubit", qubits)
        else:
            metafunc.parametrize("platform_name", platforms)
