import pytest

from qibolab._core.instruments.qblox.sequence.sweepers import Param, ParamRole, params
from qibolab._core.pulses import Pulse, Rectangular
from qibolab._core.sweeper import Parameter, Sweeper


def test_param_from_range_frequency():
    param = Param.from_range(
        (-100e6, 100e6, 10e6), Parameter.frequency, ParamRole.FREQUENCY, None, "drive"
    )
    assert param.start == int(-400e6) % (2**32)
    assert param.step == 40e6


@pytest.mark.parametrize("irange", [(500e6, 400e6, -10e6), (400e6, 500e6, 10e6)])
def test_param_from_range_invalid_frequency(irange):
    with pytest.raises(ValueError, match="IF frequency must be a float between"):
        Param.from_range(
            irange, Parameter.frequency, ParamRole.FREQUENCY, None, "drive"
        )


@pytest.mark.parametrize(
    "filter_pulses,filter_channels,expected_roles",
    [
        (None, None, [ParamRole.AMPLITUDE, ParamRole.FREQUENCY]),
        (True, None, [ParamRole.AMPLITUDE]),
        (None, True, [ParamRole.FREQUENCY]),
        (True, True, [ParamRole.AMPLITUDE, ParamRole.FREQUENCY]),
        (False, None, []),
        (None, False, []),
        (False, False, []),
        (True, False, [ParamRole.AMPLITUDE]),
        (False, True, [ParamRole.FREQUENCY]),
    ],
)
def test_params_target_filters(filter_pulses, filter_channels, expected_roles):
    pulse = Pulse(duration=40, amplitude=0.1, envelope=Rectangular())
    sweepers = [
        [
            Sweeper(
                parameter=Parameter.amplitude, range=(0.1, 0.3, 0.1), pulses=[pulse]
            ),
            Sweeper(
                parameter=Parameter.frequency, range=(0, 10e6, 1e6), channels=["drive"]
            ),
        ]
    ]
    pulse_ids = None if filter_pulses is None else [pulse.id] if filter_pulses else []
    channel_ids = (
        None if filter_channels is None else ["drive"] if filter_channels else []
    )

    result = params(sweepers, allocated=2, pulses=pulse_ids, channels=channel_ids)

    assert [param.role for param in result] == expected_roles
    assert [param.reg.number for param in result] == list(range(3, 3 + len(result)))
