import os
import pathlib
from importlib import import_module

import pytest

from qibolab import PROFILE, create_platform
from qibolab.platforms.multiqubit import MultiqubitPlatform


def pytest_addoption(parser):
    parser.addoption(
        "--platforms",
        type=str,
        action="store",
        default="qm,qblox",
        help="qpu platforms to test on",
    )
    parser.addoption(
        "--address",
        type=str,
        action="store",
        default=None,
        help="address for the QM simulator",
    )
    parser.addoption(
        "--simulation-duration",
        type=int,
        action="store",
        default=3000,
        help="simulation duration for QM simulator tests",
    )
    parser.addoption(
        "--folder",
        type=str,
        action="store",
        default=None,
        help="folder to save QM simulator test regressions",
    )


def set_platform_profile():
    os.environ[PROFILE] = str(pathlib.Path(__file__).parent / "dummy_qrc" / "profile.toml")


@pytest.fixture
def dummy_qrc():
    set_platform_profile()


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
    set_platform_profile()
    platform = create_platform(request.param[0])
    inst, _ = load_from_platform(platform, request.param[1])
    inst.connect()
    yield inst
    inst.disconnect()


def pytest_generate_tests(metafunc):
    platforms = metafunc.config.option.platforms
    platforms = [] if platforms is None else platforms.split(",")

    if "simulator" in metafunc.fixturenames:
        address = metafunc.config.option.address
        if address is None:
            pytest.skip("Skipping QM simulator tests because address was not provided.")
        else:
            duration = metafunc.config.option.simulation_duration
            folder = metafunc.config.option.folder
            metafunc.parametrize("simulator", [(address, duration)], indirect=True)
            metafunc.parametrize("folder", [folder], indirect=True)

    if metafunc.module.__name__ == "tests.test_instruments_qblox":
        set_platform_profile()
        for platform_name in platforms:
            if not isinstance(create_platform(platform_name), MultiqubitPlatform):
                pytest.skip("Skipping qblox tests because no platform is available.")

    if "instrument" in metafunc.fixturenames:
        if metafunc.module.__name__ == "tests.test_instruments_rohde_schwarz":
            metafunc.parametrize("instrument", [(p, "SGS100A") for p in platforms], indirect=True)
        elif metafunc.module.__name__ == "tests.test_instruments_erasynth":
            metafunc.parametrize("instrument", [(p, "ERA") for p in platforms], indirect=True)
        elif metafunc.module.__name__ == "tests.test_instruments_qutech":
            metafunc.parametrize("instrument", [(p, "SPI") for p in platforms], indirect=True)

    elif "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", platforms, indirect=True)

    elif "platform_name" in metafunc.fixturenames:
        set_platform_profile()
        if "qubit" in metafunc.fixturenames:
            qubits = []
            for platform_name in platforms:
                qubits.extend((platform_name, q) for q in create_platform(platform_name).qubits)
            metafunc.parametrize("platform_name,qubit", qubits)
        else:
            metafunc.parametrize("platform_name", platforms)
