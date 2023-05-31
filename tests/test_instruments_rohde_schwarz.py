import numpy as np
import pytest

from qibolab import create_platform
from qibolab.instruments.abstract import INSTRUMENTS_DATA_FOLDER

from .conftest import load_from_platform


@pytest.mark.qpu
def test_instruments_rohde_schwarz_init(instrument):
    assert instrument.is_connected == True
    assert instrument.device
    assert instrument.data_folder == INSTRUMENTS_DATA_FOLDER / instrument.tmp_folder.name.split("/")[-1]


@pytest.mark.qpu
@pytest.mark.parametrize("instrument_name", ["SGS100A"])
def test_instruments_rohde_schwarz_setup(platform_name, instrument_name):
    platform = create_platform(platform_name)
    settings = platform.settings
    instrument, instrument_settings = load_from_platform(platform, instrument_name)
    instrument.connect()
    instrument.setup(**settings["settings"], **instrument_settings)
    for parameter in instrument_settings:
        assert getattr(instrument, parameter) == instrument_settings[parameter]
    instrument.disconnect()


def instrument_set_and_test_parameter_values(instrument, parameter, values):
    for value in values:
        instrument._set_device_parameter(parameter, value)
        assert instrument.device.get(parameter) == value


@pytest.mark.qpu
def test_instruments_rohde_schwarz_set_device_paramter(instrument):
    instrument_set_and_test_parameter_values(
        instrument, f"power", np.arange(-120, 0, 10)
    )  # Max power is 25dBm but to be safe testing only until 0dBm
    instrument_set_and_test_parameter_values(instrument, f"frequency", np.arange(1e6, 12750e6, 1e9))
    """   # TODO: add attitional paramter tests
    SGS100A:
        parameter            value
    --------------------------------------------------------------------------------
    IDN                   :	{'vendor': 'Rohde&Schwarz', 'model': 'SGS100A', 'seri...
    IQ_angle              :	None
    IQ_gain_imbalance     :	None
    IQ_impairments        :	None
    IQ_state              :	None
    I_offset              :	None
    LO_source             :	None
    Q_offset              :	None
    frequency             :	None (Hz)
    phase                 :	None (deg)
    power                 :	None (dBm)
    pulsemod_source       :	None
    pulsemod_state        :	None
    ref_LO_out            :	None
    ref_osc_external_freq :	None
    ref_osc_output_freq   :	None
    ref_osc_source        :	None
    status                :	None
    timeout               :	5 (s)
    """


@pytest.mark.qpu
def test_instruments_rohde_schwarz_start_stop_disconnect(instrument):
    instrument.start()
    instrument.stop()
    instrument.disconnect()
    assert instrument.is_connected == False
