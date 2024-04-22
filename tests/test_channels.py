import pytest

from qibolab.channels import Channel, ChannelMap
from qibolab.instruments.dummy import DummyPort


def test_channel_init():
    channel = Channel("L1-test")
    assert channel.name == "L1-test"


def test_channel_errors():
    channel = Channel("L1-test", port=DummyPort("test"))
    channel.offset = 0.1
    channel.filter = {}
    # attempt to set bias higher than the allowed value
    channel.max_offset = 0.2
    with pytest.raises(ValueError):
        channel.offset = 0.3


def test_channel_map_add():
    channels = ChannelMap().add("a", "b")
    assert "a" in channels
    assert "b" in channels
    assert isinstance(channels["a"], Channel)
    assert isinstance(channels["b"], Channel)
    assert channels["a"].name == "a"
    assert channels["b"].name == "b"


def test_channel_map_setitem():
    channels = ChannelMap()
    with pytest.raises(TypeError):
        channels["c"] = "test"
    channels["c"] = Channel("c")
    assert isinstance(channels["c"], Channel)


def test_channel_map_union():
    channels1 = ChannelMap().add("a", "b")
    channels2 = ChannelMap().add("c", "d")
    channels = channels1 | channels2
    for name in ["a", "b", "c", "d"]:
        assert name in channels
        assert isinstance(channels[name], Channel)
        assert channels[name].name == name
    assert "a" not in channels2
    assert "b" not in channels2
    assert "c" not in channels1
    assert "d" not in channels1


def test_channel_map_union_update():
    channels = ChannelMap().add("a", "b")
    channels |= ChannelMap().add("c", "d")
    for name in ["a", "b", "c", "d"]:
        assert name in channels
        assert isinstance(channels[name], Channel)
        assert channels[name].name == name


@pytest.fixture
def first_qubit(platform):
    return next(iter(platform.qubits.values()))


def test_platform_lo_drive_frequency(first_qubit):
    first_qubit.drive.lo_frequency = 5.5e9
    assert first_qubit.drive.lo_frequency == 5.5e9


def test_platform_lo_readout_frequency(first_qubit):
    first_qubit.readout.lo_frequency = 7e9
    assert first_qubit.readout.lo_frequency == 7e9


def test_platform_attenuation(first_qubit):
    first_qubit.drive.attenuation = 10
    assert first_qubit.drive.attenuation == 10


def test_platform_gain(first_qubit):
    first_qubit.readout.gain = 0
    assert first_qubit.readout.gain == 0


def test_platform_bias(first_qubit):
    first_qubit.flux.offset = 0.05
    assert first_qubit.flux.offset == 0.05
