import os
import pathlib

import pytest

from qibolab import PLATFORMS, create_platform

ORIGINAL_PLATFORMS = os.environ.get(PLATFORMS, "")
TESTING_PLATFORM_NAMES = [
    "dummy_couplers",
    "qm",
    "qm_octave",
    "qblox",
    "rfsoc",
    "zurich",
]
"""Platforms used for testing without access to real instruments."""


def pytest_addoption(parser):
    parser.addoption(
        "--platform",
        type=str,
        action="store",
        default=None,
        help="qpu platform to test on",
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


def find_instrument(platform, instrument_type):
    for instrument in platform.instruments.values():
        if isinstance(instrument, instrument_type):
            return instrument
    return None


def get_instrument(platform, instrument_type):
    """Finds if an instrument of a given type exists in the given platform.

    If the platform does not have such an instrument, the corresponding
    test that asked for this instrument is skipped. This ensures that
    QPU tests are executed only on the available instruments.
    """
    instrument = find_instrument(platform, instrument_type)
    if instrument is None:
        pytest.skip(f"Skipping {instrument_type.__name__} test for {platform.name}.")
    return instrument


@pytest.fixture(scope="module", params=TESTING_PLATFORM_NAMES)
def platform(request):
    """Dummy platform to be used when there is no access to QPU.

    This fixture should be used only by tests that do are not marked
    as ``qpu``.

    Dummy platforms are defined in ``tests/dummy_qrc`` and do not
    need to be updated over time.
    """
    set_platform_profile()
    return create_platform(request.param)


@pytest.fixture(scope="module")
def connected_platform(request):
    """Platform that has access to QPU instruments.

    This fixture should be used for tests that are marked as ``qpu``.

    These platforms are defined in the folder specified by
    the ``QIBOLAB_PLATFORMS`` environment variable.
    """
    os.environ[PLATFORMS] = ORIGINAL_PLATFORMS
    name = request.config.getoption("--platform")
    platform = create_platform(name)
    platform.connect()
    yield platform
    platform.disconnect()
