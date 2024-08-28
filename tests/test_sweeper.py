import numpy as np
import pytest

from qibolab.identifier import ChannelId
from qibolab.pulses import Pulse, Rectangular
from qibolab.sweeper import ChannelParameter, Parameter, Sweeper


@pytest.mark.parametrize("parameter", Parameter)
def test_sweeper_pulses(parameter):
    pulse = Pulse(
        duration=40,
        amplitude=0.1,
        envelope=Rectangular(),
    )
    if parameter is Parameter.amplitude:
        parameter_range = np.random.rand(10)
    else:
        parameter_range = np.random.randint(10, size=10)
    if parameter in ChannelParameter:
        with pytest.raises(
            ValueError, match="Cannot create a sweeper .* without specifying channels"
        ):
            _ = Sweeper(parameter=parameter, values=parameter_range, pulses=[pulse])
    else:
        sweeper = Sweeper(parameter=parameter, values=parameter_range, pulses=[pulse])
        assert sweeper.parameter is parameter


@pytest.mark.parametrize("parameter", Parameter)
def test_sweeper_channels(parameter):
    channel = ChannelId.load("0/probe")
    parameter_range = np.random.randint(10, size=10)
    if parameter in ChannelParameter:
        sweeper = Sweeper(
            parameter=parameter, values=parameter_range, channels=[channel]
        )
        assert sweeper.parameter is parameter
    else:
        with pytest.raises(
            ValueError, match="Cannot create a sweeper .* without specifying pulses"
        ):
            _ = Sweeper(parameter=parameter, values=parameter_range, channels=[channel])


def test_sweeper_errors():
    channel = ChannelId.load("0/probe")
    pulse = Pulse(
        duration=40,
        amplitude=0.1,
        envelope=Rectangular(),
    )
    parameter_range = np.random.randint(10, size=10)
    with pytest.raises(
        ValueError,
        match="Cannot create a sweeper without specifying pulses or channels",
    ):
        Sweeper(parameter=Parameter.frequency, values=parameter_range)
    with pytest.raises(
        ValueError, match="Cannot create a sweeper by using both pulses and channels"
    ):
        Sweeper(
            parameter=Parameter.frequency,
            values=parameter_range,
            pulses=[pulse],
            channels=[channel],
        )
    with pytest.raises(
        ValueError, match="'linspace' and 'values' are mutually exclusive"
    ):
        Sweeper(
            parameter=Parameter.frequency,
            values=parameter_range,
            linspace=(0, 10, 1),
            channels=[channel],
        )
