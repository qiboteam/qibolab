import numpy as np
import pytest

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.pulses import PulseSequence
from qibolab.result import (
    AveragedIntegratedResults,
    AveragedSampleResults,
    IntegratedResults,
    SampleResults,
)
from qibolab.sweeper import Parameter, Sweeper

NSHOTS = 50
NSWEEP = 5


def execute(platform, acquisition_type, averaging_mode, sweep=False):
    qubit = next(iter(platform.qubits))

    qd_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_MZ_pulse(qubit, start=qd_pulse.finish)
    sequence = PulseSequence()
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    options = ExecutionParameters(nshots=NSHOTS, acquisition_type=acquisition_type, averaging_mode=averaging_mode)
    if sweep:
        values = np.linspace(0.1, 0.5, NSWEEP)
        sweeper1 = Sweeper(Parameter.amplitude, values, pulses=[qd_pulse])
        sweeper2 = Sweeper(Parameter.amplitude, values, pulses=[ro_pulse])
        results = platform.sweep(sequence, options, sweeper1, sweeper2)
    else:
        results = platform.execute_pulse_sequence(sequence, options)
    return results[qubit]


@pytest.mark.qpu
@pytest.mark.parametrize("sweep", [False, True])
def test_discrimination_singleshot(connected_platform, sweep):
    result = execute(connected_platform, AcquisitionType.DISCRIMINATION, AveragingMode.SINGLESHOT, sweep)
    assert isinstance(result, SampleResults)
    if sweep:
        assert result.samples.shape == (NSHOTS, NSWEEP, NSWEEP)
    else:
        assert result.samples.shape == (NSHOTS,)


@pytest.mark.qpu
@pytest.mark.parametrize("sweep", [False, True])
def test_discrimination_cyclic(connected_platform, sweep):
    result = execute(connected_platform, AcquisitionType.DISCRIMINATION, AveragingMode.CYCLIC, sweep)
    assert isinstance(result, AveragedSampleResults)
    if sweep:
        assert result.statistical_frequency.shape == (NSWEEP, NSWEEP)
    else:
        assert result.statistical_frequency.shape == tuple()


@pytest.mark.qpu
@pytest.mark.parametrize("sweep", [False, True])
def test_integration_singleshot(connected_platform, sweep):
    result = execute(connected_platform, AcquisitionType.INTEGRATION, AveragingMode.SINGLESHOT, sweep)
    assert isinstance(result, IntegratedResults)
    if sweep:
        assert result.voltage.shape == (NSHOTS, NSWEEP, NSWEEP)
    else:
        assert result.voltage.shape == (NSHOTS,)


@pytest.mark.qpu
@pytest.mark.parametrize("sweep", [False, True])
def test_integration_cyclic(connected_platform, sweep):
    result = execute(connected_platform, AcquisitionType.INTEGRATION, AveragingMode.CYCLIC, sweep)
    assert isinstance(result, AveragedIntegratedResults)
    if sweep:
        assert result.voltage.shape == (NSWEEP, NSWEEP)
    else:
        assert result.voltage.shape == tuple()
