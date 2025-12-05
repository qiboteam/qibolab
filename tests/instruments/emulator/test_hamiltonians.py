"""Testing hamiltonians functions."""

import numpy as np
import pytest

from qibolab._core.components import AcquisitionConfig, DcConfig
from qibolab._core.instruments.emulator.hamiltonians import (
    DriveEmulatorConfig,
    ModulatedDelay,
    ModulatedDrive,
    ModulatedVirtualZ,
    Qubit,
    waveform,
)
from qibolab._core.pulses.envelope import Rectangular
from qibolab._core.pulses.pulse import Delay, Pulse, VirtualZ, _PulseLike


def test_dummy_waveform():
    acq_config = AcquisitionConfig(delay=0, smearing=0)
    dc_config = DcConfig(offset=0)
    qubit = Qubit()
    assert (
        waveform(pulse=_PulseLike(), config=acq_config, qubit=qubit, sampling_rate=1)
        is None
    )
    assert (
        waveform(pulse=_PulseLike(), config=dc_config, qubit=qubit, sampling_rate=1)
        is None
    )


@pytest.mark.parametrize(
    "pulse",
    [
        Pulse(amplitude=1, duration=10, envelope=Rectangular()),
        Delay(duration=10),
        VirtualZ(phase=0.123),
    ],
)
@pytest.mark.parametrize("level", [1, 2])
def test_iq_waveform(pulse, level):
    iq_config = DriveEmulatorConfig(frequency=5e9)
    qubit = Qubit(frequency=5e9)
    modulated = waveform(pulse=pulse, config=iq_config, qubit=qubit, sampling_rate=1)
    if isinstance(pulse, Pulse):
        assert isinstance(modulated, ModulatedDrive)
        assert pytest.approx(modulated.omega) == 2 * np.pi * 5
    if isinstance(pulse, Delay):
        assert isinstance(modulated, ModulatedDelay)
    if isinstance(pulse, VirtualZ):
        assert isinstance(modulated, ModulatedVirtualZ)
        with pytest.raises(ValueError, match="VirtualZ doesn't have waveform."):
            modulated(0, 0, 0)
