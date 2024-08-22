from qibolab.identifier import ChannelType


def test_channel_type():
    assert str(ChannelType.ACQUISITION) == "acquisition"
