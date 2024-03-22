import datetime
from unittest import mock

import pytest

from qibolab.instruments.bluefors import TemperatureController

messages = [
    "4K-flange: {'temperature':3.065067, 'timestamp':1710912431.128234}",
    """50K-flange: {'temperature':35.733738, 'timestamp':1710956419.545651}
4K-flange: {'temperature':3.065067, 'timestamp':1710955431.128234}""",
]


@pytest.mark.parametrize("message", messages)
def test_message_converted_json(message):
    tc = TemperatureController("Test_Temperature_Controller", "")
    converted_message = tc.convert_to_json(message)
    assert isinstance(converted_message, dict)
    for key, value in converted_message.items():
        assert key.endswith("-flange")
        assert isinstance(value, dict)
        assert set(value.keys()) == {"temperature", "timestamp", "time"}
        assert isinstance(value["temperature"], float)
        assert isinstance(value["timestamp"], float)
        assert isinstance(value["time"], datetime.datetime)


def test_connect():
    with mock.patch("socket.socket"):
        tc = TemperatureController("Test_Temperature_Controller", "")
        assert tc.is_connected is False
        # if already connected, it should stay connected
        for _ in range(2):
            tc.connect()
            assert tc.is_connected is True


@pytest.mark.parametrize("already_connected", [True, False])
def test_disconnect(already_connected):
    with mock.patch("socket.socket"):
        tc = TemperatureController("Test_Temperature_Controller", "")
        if not already_connected:
            tc.connect()
        # if already disconnected, it should stay disconnected
        for _ in range(2):
            tc.disconnect()
            assert tc.is_connected is False


def test_continuously_read_data():
    with mock.patch(
        "qibolab.instruments.bluefors.TemperatureController.get_data",
        new=lambda _: TemperatureController.convert_to_json(messages[0]),
    ):
        tc = TemperatureController("Test_Temperature_Controller", "")
        read_temperatures = tc.read_data()
        for read_temperature in read_temperatures:
            assert read_temperature == TemperatureController.convert_to_json(
                messages[0]
            )
            break
