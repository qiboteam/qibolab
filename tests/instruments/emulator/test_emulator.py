"""Testing emulator basic functionalities."""

import numpy as np
import pytest

from qibolab._core.execution_parameters import AcquisitionType, AveragingMode
from qibolab._core.pulses.pulse import Align
from qibolab._core.sweeper import Parameter, Sweeper

NSHOTS = 1000
"""Number of shots to be used for tests."""


def test_integration_mode(platform):
    seq = platform.natives.single_qubit[0].MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    result = platform.execute(
        [seq],
        nshots=NSHOTS,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    assert result[acq_handle].shape == (NSHOTS, 2)
    pytest.approx(result[acq_handle][:, 1].mean(), abs=1e-2) == 0

    result = platform.execute(
        [seq],
        nshots=NSHOTS,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    assert result[acq_handle].shape == (2,)
    pytest.approx(result[acq_handle][1], abs=1e-2) == 0


def test_align_fail(platform):
    q0 = platform.natives.single_qubit[0]
    seq = q0.RX() | q0.MZ()
    seq += [(platform.qubits[0].drive, Align())]
    with pytest.raises(ValueError, match="Align not supported in emulator."):
        platform.execute([seq])


def test_sweepers(platform):
    q0 = platform.natives.single_qubit[0]
    seq = q0.RX() | q0.MZ()
    ch, pulse = seq[0]
    sweeper = Sweeper(
        parameter=Parameter.amplitude, values=np.array([0, 1]), pulses=[pulse]
    )
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], [[sweeper]], nshots=NSHOTS)
    assert res[acq_handle].shape == (NSHOTS, 2)

    sweeper = Sweeper(
        parameter=Parameter.frequency, values=np.array([0, 1]), channels=[ch]
    )
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], [[sweeper]], nshots=NSHOTS)
    assert res[acq_handle].shape == (NSHOTS, 2)
