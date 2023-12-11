import pytest

from qibolab.instruments.qblox.controller import QbloxController


def get_controller(platform):
    for instrument in platform.instruments.values():
        if isinstance(instrument, QbloxController):
            return instrument
    pytest.skip(f"Skipping qblox test for {platform.name}.")


@pytest.fixture(scope="module")
def controller(platform):
    return get_controller(platform)


@pytest.fixture(scope="module")
def connected_controller(connected_platform):
    return get_controller(connected_platform)
