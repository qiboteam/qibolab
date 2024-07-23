import numpy as np
import pytest

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.result import (
    AveragedIntegratedResults,
    AveragedSampleResults,
    IntegratedResults,
    SampleResults,
)
from qibolab.sweeper import Parameter, Sweeper

NSHOTS = 50
NSWEEP1 = 5
NSWEEP2 = 8


def execute(platform: Platform, acquisition_type, averaging_mode, sweep=False):
    qubit = next(iter(platform.qubits))

    qd_pulse = platform.create_RX_pulse(qubit, start=0)
    ro_pulse = platform.create_MZ_pulse(qubit, start=qd_pulse.finish)
    sequence = PulseSequence()
    sequence.append(qd_pulse)
    sequence.append(ro_pulse)

    options = ExecutionParameters(
        nshots=NSHOTS, acquisition_type=acquisition_type, averaging_mode=averaging_mode
    )
    if sweep:
        amp_values = np.arange(0.01, 0.06, 0.01)
        freq_values = np.arange(-4e6, 4e6, 1e6)
        sweeper1 = Sweeper(Parameter.bias, amp_values, qubits=[platform.qubits[qubit]])
        # sweeper1 = Sweeper(Parameter.amplitude, amp_values, pulses=[qd_pulse])
        sweeper2 = Sweeper(Parameter.frequency, freq_values, pulses=[ro_pulse])
        results = platform.execute([sequence], options, [[sweeper1], [sweeper2]])
    else:
        results = platform.execute([sequence], options)
    return results[qubit][0]


@pytest.mark.qpu
@pytest.mark.parametrize("sweep", [False, True])
def test_discrimination_singleshot(connected_platform, sweep):
    result = execute(
        connected_platform,
        AcquisitionType.DISCRIMINATION,
        AveragingMode.SINGLESHOT,
        sweep,
    )
    assert isinstance(result, SampleResults)
    if sweep:
        assert result.samples.shape == (NSHOTS, NSWEEP1, NSWEEP2)
    else:
        assert result.samples.shape == (NSHOTS,)


@pytest.mark.qpu
@pytest.mark.parametrize("sweep", [False, True])
def test_discrimination_cyclic(connected_platform, sweep):
    result = execute(
        connected_platform, AcquisitionType.DISCRIMINATION, AveragingMode.CYCLIC, sweep
    )
    assert isinstance(result, AveragedSampleResults)
    if sweep:
        assert result.statistical_frequency.shape == (NSWEEP1, NSWEEP2)
    else:
        assert result.statistical_frequency.shape == tuple()


@pytest.mark.qpu
@pytest.mark.parametrize("sweep", [False, True])
def test_integration_singleshot(connected_platform, sweep):
    result = execute(
        connected_platform, AcquisitionType.INTEGRATION, AveragingMode.SINGLESHOT, sweep
    )
    assert isinstance(result, IntegratedResults)
    if sweep:
        assert result.voltage.shape == (NSHOTS, NSWEEP1, NSWEEP2)
    else:
        assert result.voltage.shape == (NSHOTS,)


@pytest.mark.qpu
@pytest.mark.parametrize("sweep", [False, True])
def test_integration_cyclic(connected_platform, sweep):
    result = execute(
        connected_platform, AcquisitionType.INTEGRATION, AveragingMode.CYCLIC, sweep
    )
    assert isinstance(result, AveragedIntegratedResults)
    if sweep:
        assert result.voltage.shape == (NSWEEP1, NSWEEP2)
    else:
        assert result.voltage.shape == tuple()
