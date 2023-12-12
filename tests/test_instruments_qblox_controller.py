import pytest

from qibolab.instruments.qblox.controller import QbloxController

from .qblox_fixtures import connected_controller, controller


def test_init(controller: QbloxController):
    assert controller.is_connected is False
    assert type(controller.modules) == dict
    assert controller.cluster == None
    assert controller._reference_clock in ["internal", "external"]


@pytest.mark.qpu
def connect(connected_controller: QbloxController):
    connected_controller.connect()
    assert connected_controller.is_connected
    for module in connected_controller.modules.values():
        assert module.is_connected


@pytest.mark.qpu
def disconnect(connected_controller: QbloxController):
    connected_controller.connect()
    connected_controller.disconnect()
    assert connected_controller.is_connected is False
    for module in connected_controller.modules.values():
        assert module.is_connected is False
