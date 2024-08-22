import pytest
from pydantic import ValidationError

from qibolab.identifier import ChannelId, ChannelType


def test_channel_type():
    assert str(ChannelType.ACQUISITION) == "acquisition"


def test_channel_id():
    name = "1/probe"
    ch = ChannelId.load(name)
    assert ch.qubit == 1
    assert ch.channel_type is ChannelType.PROBE
    assert ch.cross is None
    assert str(ch) == name == ch.model_dump()

    chd = ChannelId.load("10/drive_cross/3")
    assert chd.qubit == 10
    assert chd.channel_type is ChannelType.DRIVE_CROSS
    assert chd.cross == "3"

    with pytest.raises(ValidationError):
        ChannelId.load("1/probe/3")

    with pytest.raises(ValueError):
        ChannelId.load("ciao/come/va/bene")
