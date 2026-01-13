import numpy as np
import pytest

from qibolab._core.pulses import Pulse, Rectangular, GaussianSquare
from qibolab._core.sweeper import Parameter, Sweeper


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
    elif parameter is Parameter.phase:
        with pytest.raises(TypeError):
            _ = Sweeper(parameter=parameter, values=parameter_range, pulses=[pulse])
    else:
        sweeper = Sweeper(parameter=parameter, values=parameter_range, pulses=[pulse])
        assert sweeper.parameter is parameter


@pytest.mark.parametrize("parameter", Parameter)
def test_sweeper_channels(parameter):
    channel = "0/probe"
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
    channel = "0/probe"
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
            channels=[channel],
        )
    with pytest.raises(ValueError, match="Amplitude"):
        Sweeper(
            parameter=Parameter.amplitude,
            range=(0, 2, 0.2),
            pulses=[pulse],
        )


def test_sweepers_equivalence():
    channel_1 = "0/probe"
    pulse_1 = Pulse(
        duration=40,
        amplitude=0.1,
        envelope=Rectangular(),
    )

    parameter_range = np.random.randint(10, size=10)

    sweeper1 = Sweeper(parameter=Parameter.frequency,
                    values=parameter_range,
                    channels=[channel_1],
    )

    sweeper2 = sweeper1.model_copy()
    assert sweeper1 == sweeper2

    # changing channel of the sweeper
    sweeper2 = sweeper1.model_copy(
            update= {"channels": '0/drive'}
    )
    assert sweeper1 != sweeper2

    # changing sweeper parameter
    sweeper2 = sweeper1.model_copy(
            update= {"parameter": Parameter.offset }
    )
    assert sweeper1 != sweeper2

    # changing sweeper range - different values
    sweeper2 = sweeper1.model_copy(
            update= {"values": np.random.randint(20,size=10)}
    )
    assert sweeper1 != sweeper2

    # changing sweeper range - different lenght
    sweeper2 = sweeper1.model_copy(
            update= {"values": np.random.randint(20,size=20)}
    )
    assert sweeper1 != sweeper2

    sweeper1 = Sweeper(parameter=Parameter.duration,
                    values=parameter_range,
                    pulses=[pulse_1],
    )

    # removing channnels field and adding pulses field
    assert sweeper1 != sweeper2

    # chaging number of pulses
    sweeper2 = sweeper1.model_copy(
            update= {"pulses": [pulse_1]*2}
    ) 
    assert sweeper1 != sweeper2

    # changing pulse parameters
    for keys, param in zip(['amplitude', 'duration', 'envelope'], [10, 10, GaussianSquare()]):
        pulse_2 = pulse_1.model_copy(update={keys:param})
        sweeper2 = sweeper1.model_copy(
                update= {"pulses": [pulse_2]}
        ) 
        assert sweeper1 != sweeper2

