import os
import pathlib
import pytest
import numpy as np
from qibolab.instruments import qblox
from qibolab.tests.utils import load_runcard, generate_pulse_sequence

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


def get_pulsar(device):
    """Initializes and setups a pulsar for testing.
    
    Args:
        device (str): 'QCM' or 'QRM'.
    """
    settings = load_runcard("tiiq")
    pulsar = getattr(qblox, f"Pulsar{device}")(**settings.get(f"{device}_init_settings"))
    pulsar.setup(**settings.get(f"{device}_settings"))
    return pulsar


@pytest.mark.parametrize("device", ["QCM", "QRM"])
def test_pulsar_init_and_setup(device):
    """Tests if Pulsars can be initialized and setup."""
    pulsar = get_pulsar(device)
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
    pulsar = get_pulsar(device)
    sequence = generate_pulse_sequence()
    waveforms, program = pulsar.translate(sequence, 0, 100)
    pulsar.close()

    modI, modQ = waveforms.get(f"modI_{pulsar.name}"), waveforms.get(f"modQ_{pulsar.name}")
    assert modI.get("index") == 0
    assert modQ.get("index") == 1
    assert_regression_array(modI.get("data"), f"{pulsar.name}_waveforms_modI.txt")
    assert_regression_array(modQ.get("data"), f"{pulsar.name}_waveforms_modQ.txt")
    assert_regression_str(program, f"{pulsar.name}_program.txt")


@pytest.mark.parametrize("device", ["QCM", "QRM"])
def test_upload_and_play_sequence(device):
    """Tests uploading and executing waveforms in pulsars."""
    import shutil
    pulsar = get_pulsar(device)
    sequence = generate_pulse_sequence()
    waveforms, program = pulsar.translate(sequence, 0, 100)

    if pulsar._connected:
        pulsar.upload(waveforms, program, "./data")
        if device == "QCM":
            pulsar.play_sequence()
        else:
            pulsar.play_sequence_and_acquire(sequence.qrm_pulses[-1])
    else:
        with pytest.raises(AttributeError):
            pulsar.upload(waveforms, program, "./data")

    pulsar.close()
    shutil.rmtree("./data")


# TODO: Test ``PulsarQRM._demodulate_and_integrate`` (requires some output from execution)