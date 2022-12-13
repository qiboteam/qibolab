import pathlib

import numpy as np

from qibolab import Platform
from qibolab.pulses import PulseSequence

REGRESSION_FOLDER = pathlib.Path(__file__).with_name("regressions")


def assert_regression_fixture(array, filename, rtol=1e-7, atol=1e-12):
    """Check array matches data inside filename.

    Args:
        array: numpy array/
        filename: fixture filename

    If filename does not exists, this function
    creates the missing file otherwise it loads
    from file and compare.
    """

    def load(filename):
        return np.loadtxt(filename)

    filename = REGRESSION_FOLDER / filename
    try:
        array_fixture = load(filename)
    except:  # pragma: no cover
        # case not tested in GitHub workflows because files exist
        np.savetxt(filename, array)
        array_fixture = load(filename)
    print(array.shape)
    print(array_fixture.shape)
    np.testing.assert_allclose(array, array_fixture, rtol=rtol, atol=atol)


# TODO: Add tests for ``platform.config``
# TODO: Test different configurations of pulse sequence executions


def test_pulse_sequence_execution_simulated_waveforms(qmsim_address):
    platform = Platform("qm")
    platform.connect(qmsim_address)
    platform.setup()

    qd_pulse = platform.create_RX_pulse(1, start=0)
    qd_pulse2 = platform.create_RX_pulse(1, start=qd_pulse.duration)
    ro_pulse = platform.create_MZ_pulse(1, start=2 * qd_pulse.duration)
    sequence = PulseSequence()
    sequence.add(qd_pulse)
    sequence.add(qd_pulse2)
    sequence.add(ro_pulse)

    result = platform.execute_pulse_sequence(sequence, nshots=1, simulation_duration=1000)
    samples = result.get_simulated_samples()
    ports = sorted(samples.con1.analog.keys())
    samples_array = np.stack([samples.con1.analog[p] for p in ports])
    assert_regression_fixture(samples_array, "qm_sequence_execution.out")
