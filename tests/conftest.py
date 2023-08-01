import os
import pathlib

import pytest

from qibolab import PLATFORMS, create_platform

ORIGINAL_PLATFORMS = os.environ.get(PLATFORMS, "")
DUMMY_PLATFORM_NAMES = ["qm", "qblox", "rfsoc", "zurich"]


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


def get_instrument(platform, instrument_type):
    for instrument in platform.instruments:
        if isinstance(instrument, instrument_type):
            return instrument
    pytest.skip(f"Skipping {instrument_type.__name__} test for {platform.name}.")


@pytest.fixture(scope="module", params=DUMMY_PLATFORM_NAMES)
def platform(request):
    set_platform_profile()
    return create_platform(request.param)


@pytest.fixture(scope="session")
def connected_platform(request):
    os.environ[PLATFORMS] = ORIGINAL_PLATFORMS
    name = request.config.getoption("--platform")
    platform = create_platform(name)
    platform.connect()
    platform.setup()
    yield platform
    platform.disconnect()


def pytest_generate_tests(metafunc):
    if "simulator" in metafunc.fixturenames:
        address = metafunc.config.option.address
        if address is None:
            pytest.skip("Skipping QM simulator tests because address was not provided.")
        else:
            duration = metafunc.config.option.simulation_duration
            folder = metafunc.config.option.folder
            metafunc.parametrize("simulator", [(address, duration)], indirect=True)
            metafunc.parametrize("folder", [folder], indirect=True)

    # if "platform" in metafunc.fixturenames:
    #    markers = {marker.name for marker in metafunc.definition.iter_markers()}
    #    if "qpu" in markers:
    #        # use real platforms
    #        os.environ[PLATFORMS] = ORIGINAL_PLATFORMS
    #        platforms = [create_platform(name) for name in platform_names]
    #    else:
    #        # use platforms under ``dummy_qrc`` folder in tests
    #        set_platform_profile()
    #        platforms = [create_platform(name) for name in dummy_platform_names]
    #    if "qubit" in metafunc.fixturenames:
    #        config = [
    #            (platform, q)
    #            for platform in platforms
    #            for q, qubit in platform.qubits.items()
    #            if qubit.drive is not None
    #        ]
    #        metafunc.parametrize("platform,qubit", config, scope="module")
    #    else:
    #        metafunc.parametrize("platform", platforms, scope="session")
