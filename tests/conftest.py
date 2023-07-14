import os
import pathlib

import pytest

from qibolab import PLATFORMS, create_platform

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


# @pytest.fixture(scope="session")
# def platforms():
#    set_platform_profile()
#    #platform_names = os.environ.get("TEST_PLATFORMS")
#    platform_names = "qblox"
#    platform_names = [] if platform_names is None else platform_names.split(",")
#    return [create_platform(name) for name in platform_names]


def pytest_generate_tests(metafunc):
    is_qpu = "qpu" in {m.name for m in metafunc.definition.iter_markers()}

    platform_names = metafunc.config.option.platforms
    platform_names = [] if platform_names is None else platform_names.split(",")

    if "simulator" in metafunc.fixturenames:
        address = metafunc.config.option.address
        if address is None:
            pytest.skip("Skipping QM simulator tests because address was not provided.")
        else:
            duration = metafunc.config.option.simulation_duration
            folder = metafunc.config.option.folder
            metafunc.parametrize("simulator", [(address, duration)], indirect=True)
            metafunc.parametrize("folder", [folder], indirect=True)

    # if "instrument" in metafunc.fixturenames:
    #    if metafunc.module.__name__ == "tests.test_instruments_rohde_schwarz":
    #        metafunc.parametrize("instrument", [(p, "SGS100A") for p in platforms], indirect=True)
    #    elif metafunc.module.__name__ == "tests.test_instruments_erasynth":
    #        metafunc.parametrize("instrument", [(p, "ERA") for p in platforms], indirect=True)
    #    elif metafunc.module.__name__ == "tests.test_instruments_qutech":
    #        metafunc.parametrize("instrument", [(p, "SPI") for p in platforms], indirect=True)

    if "backend" in metafunc.fixturenames:
        set_platform_profile()
        metafunc.parametrize("backend", platforms, indirect=True)

    elif "platform_name" in metafunc.fixturenames:
        set_platform_profile()
        metafunc.parametrize("platform_name", platform_names)

    elif "platform" in metafunc.fixturenames:
        set_platform_profile()
        platforms = [create_platform(name) for name in platform_names]
        metafunc.parametrize("platform", platforms, scope="session")
