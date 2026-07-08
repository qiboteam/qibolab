import os
import pathlib
from collections.abc import Callable, Iterator
from typing import Optional

import numpy as np
import numpy.typing as npt
import pytest

from qibolab import AcquisitionType, AveragingMode, create_platform
from qibolab._core.platform.load import PLATFORMS_PATH
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers, Parameter, Sweeper

TESTING_PLATFORM_NAMES = ["dummy"]
"""Platforms used for testing without access to real instruments."""


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


def set_platform_profile():
    """Point platforms environment to the ``tests/dummy_qrc`` folder."""
    os.environ[PLATFORMS_PATH] = str(pathlib.Path(__file__).parent / "dummy_qrc")


@pytest.fixture
def dummy_qrc():
    set_platform_profile()


@pytest.fixture
def emulators():
    os.environ[PLATFORMS_PATH] = str(pathlib.Path(__file__).parent / "emulators")


@pytest.fixture(scope="module", params=TESTING_PLATFORM_NAMES)
def platform(request):
    """Dummy platform to be used for testing.

    This platform is only supposed to generate random results. Specific testing
    platforms may have different features, but in general the values provided should not
    be trusted as proxies of any meaningful execution.
    """
    set_platform_profile()
    return create_platform(request.param)


Execution = Callable[
    [AcquisitionType, AveragingMode, int, Optional[list[ParallelSweepers]]], npt.NDArray
]


@pytest.fixture
def execute() -> Iterator[Execution]:
    platform = create_platform("dummy")
    platform.connect()

    def wrapped(
        acquisition_type: AcquisitionType,
        averaging_mode: AveragingMode,
        nshots: int = 1000,
        sweepers: list[ParallelSweepers] | None = None,
        sequence: PulseSequence | None = None,
        target: int | None = None,
    ) -> npt.NDArray:
        options = dict(
            nshots=nshots,
            acquisition_type=acquisition_type,
            averaging_mode=averaging_mode,
        )

        qubit = next(iter(platform.qubits.values()))
        natives = platform.natives.single_qubit[0]

        if sequence is None:
            qd_seq = natives.RX()
            probe_seq = natives.MZ()
            probe_pulse = probe_seq[0][1].probe
            acq = probe_seq[0][1].acquisition
            wrapped.acquisition_duration = acq.duration
            sequence = PulseSequence()
            sequence.concatenate(qd_seq)
            sequence.concatenate(probe_seq)
            if sweepers is None:
                sweeper1 = Sweeper(
                    parameter=Parameter.offset,
                    range=(0.01, 0.06, 0.01),
                    channels=[qubit.flux],
                )
                sweeper2 = Sweeper(
                    parameter=Parameter.amplitude,
                    range=(0, 0.8, 0.1),
                    pulses=[probe_pulse],
                )
                sweepers = [[sweeper1], [sweeper2]]
            if target is None:
                target = acq.id

        # default target and sweepers only supported for default sequence
        assert target is not None
        assert sweepers is not None

        results = platform.execute([sequence], sweepers, **options)
        return results[target]

    yield wrapped
    platform.disconnect()
