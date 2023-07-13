import os
import pathlib

import pytest

from qibolab import PLATFORMS

# from importlib import import_module


# from qibolab.instruments.qblox.controller import QbloxController


def pytest_addoption(parser):
    parser.addoption(
        "--platforms",
        type=str,
        action="store",
        default="qm,qblox,rfsoc,zurich",
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
    os.environ[PLATFORMS] = str(pathlib.Path(__file__).parent / "dummy_qrc")


@pytest.fixture
def dummy_qrc():
    set_platform_profile()


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
        metafunc.parametrize("platform_name", platforms)
