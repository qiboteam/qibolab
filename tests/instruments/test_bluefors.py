from unittest import mock

import pytest
import yaml

from qibolab.instruments.bluefors import TemperatureController

messages = [
    "4K-flange: {'temperature':3.065067, 'timestamp':1710912431.128234}",
    """50K-flange: {'temperature':35.733738, 'timestamp':1710956419.545651}
4K-flange: {'temperature':3.065067, 'timestamp':1710955431.128234}""",
]


def test_connect():
    with mock.patch("socket.socket"):
        tc = TemperatureController(name="Test_Temperature_Controller", address="")
        # if already connected, it should stay connected
        for _ in range(2):
            tc.connect()


@pytest.mark.parametrize("already_connected", [True, False])
def test_disconnect(already_connected):
    with mock.patch("socket.socket"):
        tc = TemperatureController(name="Test_Temperature_Controller", address="")
        if not already_connected:
            tc.connect()
        # if already disconnected, it should stay disconnected
        for _ in range(2):
            tc.disconnect()


def test_continuously_read_data():
    with mock.patch(
        "qibolab.instruments.bluefors.TemperatureController.get_data",
        new=lambda _: yaml.safe_load(messages[0]),
    ):
        tc = TemperatureController(name="Test_Temperature_Controller", address="")
        read_temperatures = tc.read_data()
        for read_temperature in read_temperatures:
            assert read_temperature == yaml.safe_load(messages[0])
            break
