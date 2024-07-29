from unittest.mock import Mock

import pytest

from qibolab.instruments.qblox.port import QbloxOutputPort


def test_set_attenuation(caplog):
    module = Mock()
    # module.device = None
    port = QbloxOutputPort(module, 17)

    port.attenuation = 40
    assert port.attenuation == 40

    port.attenuation = 40.1
    assert port.attenuation == 40

    port.attenuation = 65
    assert port.attenuation == 60
    assert "attenuation needs to be between 0 and 60 dB" in caplog.messages[0]
    caplog.clear()

    port.attenuation = -10
    assert port.attenuation == 0
    assert "attenuation needs to be between 0 and 60 dB" in caplog.messages[0]
    caplog.clear()

    with pytest.raises(ValueError, match="Invalid"):
        port.attenuation = "something"
