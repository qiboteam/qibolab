import pathlib
import yaml
import pytest
from qibolab.platforms.qbloxplatform import QBloxPlatform
from qibolab.tests.utils import generate_pulse_sequence


def test_qbloxplatform_init():
    runcard = pathlib.Path(__file__).parent.parent / "runcards" / "tiiq.yml"
    with open(runcard, "r") as file:
        settings = yaml.safe_load(file)
    platform = QBloxPlatform("tiiq", runcard)
    settings = settings.get("settings")
    platform.data_folder == settings.get("data_folder")
    platform.hardware_avg == settings.get("hardware_avg")
    platform.sampling_rate == settings.get("sampling_rate")
    platform.software_averages == settings.get("software_averages")
    platform.repetition_duration == settings.get("repetition_duration")
    platform.resonator_frequency == settings.get("resonator_frequency")
    platform.qubit_frequency == settings.get("qubit_frequency")
    platform.pi_pulse_gain == settings.get("pi_pulse_gain")
    platform.pi_pulse_amplitude == settings.get("pi_pulse_amplitude")
    platform.pi_pulse_frequency == settings.get("pi_pulse_frequency")
    platform.max_readout_voltage == settings.get("max_readout_voltage")
    platform.min_readout_voltage == settings.get("min_readout_voltage")
    platform.delay_between_pulses == settings.get("delay_between_pulses")
    platform.delay_before_readout == settings.get("delay_before_readout")
    # test setter
    platform.software_averages = 5
    platform.software_averages == 5
    with pytest.raises(RuntimeError):
        platform._check_connected()

# TODO: Test ``AbstractPlatform.run_calibration`` method

def test_qbloxplatform_connect():
    runcard = pathlib.Path(__file__).parent.parent / "runcards" / "tiiq.yml"
    platform = QBloxPlatform("tiiq", runcard)
    try:
        platform.connect()
        from qibolab.instruments.qblox import PulsarQCM, PulsarQRM
        from qibolab.instruments.rohde_schwarz import SGS100A
        assert isinstance(platform.qcm, PulsarQCM)
        assert isinstance(platform.qrm, PulsarQRM)
        assert isinstance(platform.LO_qcm, SGS100A)
        assert isinstance(platform.LO_qrm, SGS100A)
        platform.disconnect()
    except RuntimeError:
        with pytest.raises(RuntimeError):
            platform.connect()


@pytest.mark.xfail
def test_qbloxplatform_start_stop():
    runcard = pathlib.Path(__file__).parent.parent / "runcards" / "tiiq.yml"
    platform = QBloxPlatform("tiiq", runcard)
    platform.connect()
    platform.setup()
    platform.start()
    platform.stop()
    platform.disconnect()


def test_qbloxplatform_execute_error():
    runcard = pathlib.Path(__file__).parent.parent / "runcards" / "tiiq.yml"
    platform = QBloxPlatform("tiiq", runcard)
    sequence = generate_pulse_sequence()
    with pytest.raises(RuntimeError):
        results = platform(sequence, nshots=100)


@pytest.mark.xfail
def test_qbloxplatform_execute():
    runcard = pathlib.Path(__file__).parent.parent / "runcards" / "tiiq.yml"
    platform = QBloxPlatform("tiiq", runcard)
    platform.connect()
    platform.setup()
    platform.start()
    sequence = generate_pulse_sequence()
    results = platform(sequence, nshots=100)
    platform.stop()
    platform.disconnect()