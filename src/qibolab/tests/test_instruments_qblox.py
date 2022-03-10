import os
import pathlib
import pytest
import yaml
import numpy as np
from qibolab.instruments import qblox

REGRESSION_FOLDER = pathlib.Path(__file__).with_name("regressions")


def assert_regression_array(array, filename):
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


def assert_regression_str(text, filename):
    """Check string matches data inside filename.

    Same as ``assert_regression_array`` but for string.
    """
    filedir = REGRESSION_FOLDER / filename
    if os.path.exists(filedir):
        with open(filedir, "r") as file:
            target = file.read()
        assert text == target
    else:
        with open(filedir, "w") as file:
            file.write(text)


def load_runcard(name):
    runcard = pathlib.Path(__file__).parent.parent / "runcards" / f"{name}.yml"
    with open(runcard, "r") as file:
        settings = yaml.safe_load(file)
    return settings


@pytest.mark.parametrize("device", ["QCM", "QRM"])
def test_pulsar_init_and_setup(device):
    """Tests if Pulsars can be initialized and setup."""
    settings = load_runcard("tiiq")
    pulsar = getattr(qblox, f"Pulsar{device}")(**settings.get(f"{device}_init_settings"))
    pulsar.setup(**settings.get(f"{device}_settings"))
    pulsar.close()


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
    assert_regression_array(modI.get("data"), "single_pulse_waveform_modI.txt")
    assert_regression_array(modQ.get("data"), "single_pulse_waveform_modQ.txt")


@pytest.mark.parametrize("device", ["QCM", "QRM"])
def test_translate(device):
    """Tests ``generate_waveforms`` and ``generate_program``."""
    from qibolab.pulses import Pulse, ReadoutPulse
    from qibolab.pulse_shapes import Gaussian, Rectangular
    from qibolab.circuit import PulseSequence
    sequence = PulseSequence()
    sequence.add(Pulse(start=0,
                       frequency=200000000.0,
                        amplitude=0.3,
                        duration=60,
                        phase=0,
                        shape=Gaussian(60 / 5)))
    sequence.add(Pulse(start=65,
                       frequency=200000000.0,
                       amplitude=0.8,
                       duration=25,
                       phase=0,
                       shape=Gaussian(25 / 5)))
    sequence.add(ReadoutPulse(start=90,
                              frequency=20000000.0,
                              amplitude=0.5,
                              duration=3000,
                              phase=0,
                              shape=Rectangular()))

    settings = load_runcard("tiiq")
    pulsar = getattr(qblox, f"Pulsar{device}")(**settings.get(f"{device}_init_settings"))
    pulsar.setup(**settings.get(f"{device}_settings"))
    waveforms, program = pulsar.translate(sequence, 0, 100)
    pulsar.close()

    modI, modQ = waveforms.get(f"modI_{pulsar.name}"), waveforms.get(f"modQ_{pulsar.name}")
    assert modI.get("index") == 0
    assert modQ.get("index") == 1
    assert_regression_array(modI.get("data"), f"{pulsar.name}_waveforms_modI.txt")
    assert_regression_array(modQ.get("data"), f"{pulsar.name}_waveforms_modQ.txt")
    assert_regression_str(program, f"{pulsar.name}_program.txt")
