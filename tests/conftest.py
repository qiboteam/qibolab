import os
import pathlib

import pytest

from qibolab import PLATFORMS, create_platform

ORIGINAL_PLATFORMS = os.environ.get(PLATFORMS, "")


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
    """Point platforms environment to the ``tests/dummy_qrc`` folder."""
    os.environ[PLATFORMS] = str(pathlib.Path(__file__).parent / "dummy_qrc")


@pytest.fixture
def dummy_qrc():
    set_platform_profile()


def get_instrument(platform, instrument_type):
    for instrument in platform.instruments:
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
        markers = {marker.name for marker in metafunc.definition.iter_markers()}
        if "qpu" in markers:
            # use real platforms
            os.environ[PLATFORMS] = ORIGINAL_PLATFORMS
            platforms = [create_platform(name) for name in platform_names]
            metafunc.parametrize("platform", platforms, scope="module")
        else:
            # use platforms under ``dummy_qrc`` folder in tests
            set_platform_profile()
            platforms = [create_platform(name) for name in dummy_platform_names]
            metafunc.parametrize("platform", platforms, scope="module")
