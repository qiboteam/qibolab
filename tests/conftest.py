import os
import pathlib

import pytest

from qibolab import PLATFORMS


def pytest_addoption(parser):
    parser.addoption(
        "--platforms",
        type=str,
        action="store",
        default=None,
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


def get_instrument(platform, instrument_type):
    for instrument in platform.instruments.values():
        if isinstance(instrument, instrument_type):
            return instrument
    pytest.skip(f"Skipping {instrument_type.__name__} test for {platform.name}.")


def pytest_generate_tests(metafunc):
    platform_names = metafunc.config.option.platforms
    platform_names = [] if platform_names is None else platform_names.split(",")
    dummy_platform_names = ["qm", "qblox", "rfsoc", "zurich"]

    if "simulator" in metafunc.fixturenames:
        address = metafunc.config.option.address
        if address is None:
            pytest.skip("Skipping QM simulator tests because address was not provided.")
        else:
            duration = metafunc.config.option.simulation_duration
            folder = metafunc.config.option.folder
            metafunc.parametrize("simulator", [(address, duration)], indirect=True)
            metafunc.parametrize("folder", [folder], indirect=True)

    if "platform" in metafunc.fixturenames:
        platforms = [create_platform(name) for name in platform_names]
        metafunc.parametrize("platforms", platforms)

    # TODO: Change this to platform and use marker qpu to distinguish
    elif "dummy_platform" in metafunc.fixturenames:
        set_platform_profile()
        platforms = [create_platform(name) for name in dummy_platform_names]
        metafunc.parametrize("dummy_platform", platforms)


# TODO: Continue removing platform_name from ``qutech`` onwards
