from importlib import import_module

import pytest

from qibolab.platform import Platform
from qibolab.platforms.multiqubit import MultiqubitPlatform


def pytest_addoption(parser):
    parser.addoption(
        "--platforms",
        type=str,
        action="store",
        default="qili1q_os2,qw5q_gold",
        help="qpu platforms to test on",
    )


def load_from_platform(platform, name):
    """Loads instrument from platform, if it is available.

    Useful only for testing :class:`qibolab.platforms.multiqubit.MultiqubitPlatform`.
    """
    if not isinstance(platform, MultiqubitPlatform):
        pytest.skip(f"Skipping MultiqubitPlatform test for {platform}.")
    settings = platform.settings
    for instrument in settings["instruments"].values():
        if instrument["class"] == name:
            lib = instrument["lib"]
            i_class = instrument["class"]
            address = instrument["address"]
            InstrumentClass = getattr(import_module(f"qibolab.instruments.{lib}"), i_class)
            return InstrumentClass(name, address), instrument["settings"]
    pytest.skip(f"Skip {name} test as it is not included in the tested platforms.")


@pytest.fixture(scope="module")
def instrument(request):
    platform = Platform(request.param[0])
    inst, _ = load_from_platform(platform, request.param[1])
    inst.connect()
    yield inst
    inst.disconnect()


def pytest_generate_tests(metafunc):
    platforms = metafunc.config.option.platforms
    platforms = [] if platforms is None else platforms.split(",")

    if metafunc.module.__name__ == "tests.test_instruments_qblox":
        for platform_name in platforms:
            if not isinstance(Platform(platform_name), MultiqubitPlatform):
                pytest.skip("Skipping qblox tests because no platform is available.")

    if "instrument" in metafunc.fixturenames:
        if metafunc.module.__name__ == "tests.test_instruments_rohde_schwarz":
            metafunc.parametrize("instrument", [(p, "SGS100A") for p in platforms], indirect=True)
        if metafunc.module.__name__ == "tests.test_instruments_qutech":
            metafunc.parametrize("instrument", [(p, "SPI") for p in platforms], indirect=True)

    elif "platform_name" in metafunc.fixturenames:
        if "qubit" in metafunc.fixturenames:
            qubits = []
            for platform_name in platforms:
                if platform_name == "qw5q_gold":
                    # TODO: Find a better way to handle this instead of hardcoding
                    # exclude witness qubit 5 because it is not connected to drive channel
                    qubits.extend((platform_name, q) for q in range(5))
                else:
                    qubits.extend((platform_name, q) for q in Platform(platform_name).qubits)

            metafunc.parametrize("platform_name,qubit", qubits)
        else:
            metafunc.parametrize("platform_name", platforms)
