import pytest

from qibolab.designs.channels import Channel, ChannelMap


def test_channel_init():
    channel = Channel("L1-test")
    channel.ports = [("c1", 0), ("c2", 1)]
    assert channel.name == "L1-test"
    assert channel.local_oscillator is None


def test_channel_errors():
    channel = Channel("L1-test")
    channel.ports = [("c1", 0), ("c2", 1)]
    with pytest.raises(TypeError):
        channel.bias = "test"
    channel.bias = 0.1
    with pytest.raises(TypeError):
        channel.filter = "test"
    channel.filter = {}
    # attempt to set bias higher than the allowed value
    channel.max_bias = 0.2
    with pytest.raises(ValueError):
        channel.bias = 0.3


def test_channel_map_from_names():
    channels = ChannelMap.from_names("a", "b")
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
    channels1 = ChannelMap.from_names("a", "b")
    channels2 = ChannelMap.from_names("c", "d")
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
    channels = ChannelMap.from_names("a", "b")
    channels |= ChannelMap.from_names("c", "d")
    for name in ["a", "b", "c", "d"]:
        assert name in channels
        assert isinstance(channels[name], Channel)
        assert channels[name].name == name
