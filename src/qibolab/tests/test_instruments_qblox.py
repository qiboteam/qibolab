import os
import pathlib
import pytest
import yaml
import numpy as np
from qibolab.instruments import qblox

REGRESSION_FOLDER = pathlib.Path(__file__).with_name("regressions")


def assert_regression_fixture(array, filename):
    """Check array matches data inside filename.

    If filename does not exists, this function
    creates the missing file otherwise it loads
    from file and compares.
    
    Args:
        array: numpy array
        filename: target array filename
    """
    filedir = REGRESSION_FOLDER / filename
    if os.path.exists(filedir):
        target = np.loadtxt(filedir)
        np.testing.assert_allclose(array, target)
    else:
        np.savetxt(filedir, array)


def load_runcard(name):
    runcard = pathlib.Path(__file__).parent.parent / "runcards" / f"{name}.yml"
    with open(runcard, "r") as file:
        settings = yaml.safe_load(file)
    return settings


@pytest.mark.xfail
@pytest.mark.parametrize("device", ["QCM", "QRM"])
def test_pulsar_init(device):
    settings = load_runcard("tiiq")
    pulsar = getattr(qblox, f"Pulsar{device}")(**settings.get(f"{device}_init_settings"))


def test_pulsar_setup():
    # TODO: Complete this
    pass


def test_translate_single_pulse():
    from qibolab.pulses import Pulse
    from qibolab.pulse_shapes import Gaussian
    pulse = Pulse(start=0,
                  frequency=200000000.0,
                  amplitude=0.3,
                  duration=60,
                  phase=0,
                  shape=Gaussian(60 / 5))
    waveform = qblox.GenericPulsar._translate_single_pulse(pulse)
    modI, modQ = waveform.get("modI"), waveform.get("modQ")
    assert modI.get("index") == 0
    assert modQ.get("index") == 1
    assert_regression_fixture(modI.get("data"), "single_pulse_waveform_modI.txt")
    assert_regression_fixture(modQ.get("data"), "single_pulse_waveform_modQ.txt")


@pytest.mark.parametrize("device", ["QCM", "QRM"])
def test_generate_waveforms(device):
    from qibolab.pulses import Pulse
    from qibolab.pulse_shapes import Gaussian
    pulses = [Pulse(start=0,
                    frequency=200000000.0,
                    amplitude=0.3,
                    duration=60,
                    phase=0,
                    shape=Gaussian(60 / 5)),
              Pulse(start=65,
                    frequency=200000000.0,
                    amplitude=0.8,
                    duration=20,
                    phase=0,
                    shape=Gaussian(20 / 5))]

    settings = load_runcard("tiiq")
    pulsar = getattr(qblox, f"Pulsar{device}")(**settings.get(f"{device}_init_settings"))
    waveforms = pulsar.generate_waveforms(pulses)
    modI, modQ = waveforms.get(f"modI_{pulsar.name}"), waveforms.get(f"modQ_{pulsar.name}")
    assert modI.get("index") == 0
    assert modQ.get("index") == 1
    assert_regression_fixture(modI.get("data"), f"generate_waveforms_modI_{pulsar.name}.txt")
    assert_regression_fixture(modQ.get("data"), f"generate_waveforms_modQ_{pulsar.name}.txt")
