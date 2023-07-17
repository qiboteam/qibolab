import numpy as np
import pytest

from qibolab.instruments.rohde_schwarz import SGS100A

from .conftest import get_instrument


@pytest.fixture(scope="module")
def instrument(platform):
    return get_instrument(platform, SGS100A)


@pytest.mark.qpu
def test_instruments_rohde_schwarz_init(instrument):
    instrument.connect()
    assert instrument.is_connected
    assert instrument.device
    instrument.disconnect()


@pytest.mark.qpu
def test_instruments_rohde_schwarz_setup(instrument):
    instrument.connect()
    instrument.setup(frequency=5e9, power=0)
    assert instrument.frequency == 5e9
    assert instrument.power == 0
    instrument.disconnect()


def set_and_test_parameter_values(instrument, parameter, values):
    for value in values:
        instrument._set_device_parameter(parameter, value)
        assert instrument.device.get(parameter) == value


@pytest.mark.qpu
def test_instruments_rohde_schwarz_set_device_paramter(instrument):
    instrument.connect()
    set_and_test_parameter_values(
        instrument, f"power", np.arange(-120, 0, 10)
    )  # Max power is 25dBm but to be safe testing only until 0dBm
    set_and_test_parameter_values(instrument, f"frequency", np.arange(1e6, 12750e6, 1e9))
    instrument.disconnect()
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
    instrument.connect()
    assert instrument.is_connected
    instrument.start()
    instrument.stop()
    instrument.disconnect()
    assert not instrument.is_connected
