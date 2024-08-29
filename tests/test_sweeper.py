import numpy as np
import pytest

from qibolab.identifier import ChannelId
from qibolab.pulses import Pulse, Rectangular
from qibolab.sweeper import Parameter, Sweeper


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
    if parameter in Parameter.channels():
        with pytest.raises(ValueError, match="channels"):
            _ = Sweeper(parameter=parameter, values=parameter_range, pulses=[pulse])
    else:
        sweeper = Sweeper(parameter=parameter, values=parameter_range, pulses=[pulse])
        assert sweeper.parameter is parameter


@pytest.mark.parametrize("parameter", Parameter)
def test_sweeper_channels(parameter):
    channel = ChannelId.load("0/probe")
    parameter_range = np.random.randint(10, size=10)
    if parameter in Parameter.channels():
        sweeper = Sweeper(
            parameter=parameter, values=parameter_range, channels=[channel]
        )
        assert sweeper.parameter is parameter
    else:
        with pytest.raises(ValueError, match="pulses"):
            _ = Sweeper(parameter=parameter, values=parameter_range, channels=[channel])


def test_sweeper_errors():
    channel = ChannelId.load("0/probe")
    pulse = Pulse(
        duration=40,
        amplitude=0.1,
        envelope=Rectangular(),
    )
    parameter_range = np.random.randint(10, size=10)
    with pytest.raises(ValueError, match="(?=.*pulses)(?=.*channels)"):
        Sweeper(parameter=Parameter.frequency, values=parameter_range)
    with pytest.raises(ValueError, match="(?=.*pulses)(?=.*channels)"):
        Sweeper(
            parameter=Parameter.frequency,
            values=parameter_range,
            pulses=[pulse],
            channels=[channel],
        )
    with pytest.raises(ValueError, match="(?=.*range)(?=.*values)"):
        Sweeper(
            parameter=Parameter.frequency,
            values=parameter_range,
            range=(0, 10, 1),
            channels=[channel],
        )
    with pytest.raises(ValueError, match="(?=.*range)(?=.*values)"):
        Sweeper(
            parameter=Parameter.frequency,
            channels=[channel],
        )
    with pytest.raises(
        ValueError, match="Amplitude sweeper cannot have values larger than 1."
    ):
        Sweeper(
            parameter=Parameter.amplitude,
            range=(0, 2, 0.2),
            pulses=[pulse],
        )
