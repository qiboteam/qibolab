import os
import pathlib
from collections.abc import Callable
from typing import Optional

import numpy as np
import numpy.typing as npt
import pytest

from qibolab import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
    Platform,
    create_platform,
)
from qibolab.platform.load import PLATFORMS
from qibolab.sequence import PulseSequence
from qibolab.sweeper import ParallelSweepers, Parameter, Sweeper

ORIGINAL_PLATFORMS = os.environ.get(PLATFORMS, "")
TESTING_PLATFORM_NAMES = ["dummy"]
"""Platforms used for testing without access to real instruments."""


def pytest_addoption(parser):
    parser.addoption(
        "--platform",
        type=str,
        action="store",
        default=None,
        help="qpu platform to test on",
    )
    parser.addoption(
        "--address",
        type=str,
        action="store",
        default=None,
        help="address for the QM simulator",
    )
    parser.addoption(
        "--simulation-duration",
        type=int,
        action="store",
        default=3000,
        help="simulation duration for QM simulator tests",
    )
    parser.addoption(
        "--folder",
        type=str,
        action="store",
        default=None,
        help="folder to save QM simulator test regressions",
    )


def set_platform_profile():
    """Point platforms environment to the ``tests/dummy_qrc`` folder."""
    os.environ[PLATFORMS] = str(pathlib.Path(__file__).parent / "dummy_qrc")


@pytest.fixture
def dummy_qrc():
    set_platform_profile()


@pytest.fixture
def emulators():
    os.environ[PLATFORMS] = str(pathlib.Path(__file__).parent / "emulators")


def find_instrument(platform, instrument_type):
    for instrument in platform.instruments.values():
        if isinstance(instrument, instrument_type):
            return instrument
    return None


def get_instrument(platform, instrument_type):
    """Finds if an instrument of a given type exists in the given platform.

    If the platform does not have such an instrument, the corresponding
    test that asked for this instrument is skipped. This ensures that
    QPU tests are executed only on the available instruments.
    """
    instrument = find_instrument(platform, instrument_type)
    if instrument is None:
        pytest.skip(f"Skipping {instrument_type.__name__} test for {platform.name}.")
    return instrument


@pytest.fixture(scope="module", params=TESTING_PLATFORM_NAMES)
def platform(request):
    """Dummy platform to be used when there is no access to QPU.

    This fixture should be used only by tests that do are not marked
    as ``qpu``.

    Dummy platforms are defined in ``tests/dummy_qrc`` and do not
    need to be updated over time.
    """
    set_platform_profile()
    return create_platform(request.param)


@pytest.fixture(scope="module")
def connected_platform(request):
    """Platform that has access to QPU instruments.

    This fixture should be used for tests that are marked as ``qpu``.

    These platforms are defined in the folder specified by
    the ``QIBOLAB_PLATFORMS`` environment variable.
    """
    os.environ[PLATFORMS] = ORIGINAL_PLATFORMS
    name = request.config.getoption("--device", default="dummy")
    platform = create_platform(name)
    platform.connect()
    yield platform
    platform.disconnect()


Execution = Callable[
    [AcquisitionType, AveragingMode, int, Optional[list[ParallelSweepers]]], npt.NDArray
]


@pytest.fixture
def execute(connected_platform: Platform) -> Execution:
    def wrapped(
        acquisition_type: AcquisitionType,
        averaging_mode: AveragingMode,
        nshots: int = 1000,
        sweepers: Optional[list[ParallelSweepers]] = None,
        sequence: Optional[PulseSequence] = None,
        target: Optional[tuple[int, int]] = None,
    ) -> npt.NDArray:
        options = ExecutionParameters(
            nshots=nshots,
            acquisition_type=acquisition_type,
            averaging_mode=averaging_mode,
        )

        qubit = next(iter(connected_platform.qubits.values()))
        natives = connected_platform.natives.single_qubit[0]

        if sequence is None:
            qd_seq = natives.RX.create_sequence()
            probe_seq = natives.MZ.create_sequence()
            probe_pulse = probe_seq[0][1]
            acq = probe_seq[1][1]
            wrapped.acquisition_duration = acq.duration
            sequence = PulseSequence()
            sequence.concatenate(qd_seq)
            sequence.concatenate(probe_seq)
            if sweepers is None:
                freq_values = np.arange(-4e6, 4e6, 1e6)
                sweeper1 = Sweeper(
                    parameter=Parameter.offset,
                    range=(0.01, 0.06, 0.01),
                    channels=[qubit.flux.name],
                )
                sweeper2 = Sweeper(
                    parameter=Parameter.amplitude,
                    values=freq_values,
                    pulses=[probe_pulse],
                )
                sweepers = [[sweeper1], [sweeper2]]
            if target is None:
                target = (acq.id, 0)

        # default target and sweepers only supported for default sequence
        assert target is not None
        assert sweepers is not None

        results = connected_platform.execute([sequence], options, sweepers)
        return results[target[0]][target[1]]

    return wrapped
