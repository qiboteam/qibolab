import pytest

from qibolab._core.instruments.qblox.sequence.sweepers import Param, ParamRole
from qibolab._core.sweeper import Parameter


def test_param_from_range_frequency():
    param = Param.from_range(
        (-100e6, 100e6, 10e6), Parameter.frequency, ParamRole.FREQUENCY, None, "drive"
    )
    assert param.start == int(-400e6) % (2**32)
    assert param.step == 40e6


@pytest.mark.parametrize(
    ("kind", "role"),
    [(Parameter.offset, ParamRole.OFFSET), (Parameter.amplitude, ParamRole.AMPLITUDE)],
)
def test_param_from_range_converts_once(kind, role):
    param = Param.from_range((0.1, 0.3, 0.1), kind, role, None, "drive")
    assert param.start == 3276
    assert param.step == 3276


@pytest.mark.parametrize("irange", [(500e6, 400e6, -10e6), (400e6, 500e6, 10e6)])
def test_param_from_range_invalid_frequency(irange):
    with pytest.raises(ValueError, match="IF frequency must be a float between"):
        Param.from_range(
            irange, Parameter.frequency, ParamRole.FREQUENCY, None, "drive"
        )
