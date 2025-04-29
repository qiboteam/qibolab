import pytest

from qibolab import AcquisitionType, create_platform
from qibolab._core.platform.platform import Platform
from qibolab._core.pulses import Delay, GaussianSquare, Pulse
from qibolab._core.sequence import PulseSequence

SWEPT_POINTS = 5


@pytest.fixture
def platform() -> Platform:
    return create_platform("dummy")


def test_dummy_initialization(platform: Platform):
    platform.connect()
    platform.disconnect()


def test_dummy_execute_coupler_pulse(platform: Platform):
    sequence = PulseSequence()

    channel = platform.coupler(0)[1].flux
    pulse = Pulse(
        duration=30,
        amplitude=0.05,
        envelope=GaussianSquare(risefall=5, sigma=0.15),
    )
    sequence.append((channel, pulse))

    _ = platform.execute([sequence], nshots=None)


def test_dummy_execute_pulse_sequence_couplers():
    platform = create_platform("dummy")
    sequence = PulseSequence()

    natives = platform.natives
    cz = natives.two_qubit[(1, 2)].CZ()

    sequence.concatenate(cz)
    sequence.append((platform.qubits[0].probe, Delay(duration=40)))
    sequence.append((platform.qubits[2].probe, Delay(duration=40)))
    sequence.concatenate(natives.single_qubit[0].MZ())
    sequence.concatenate(natives.single_qubit[2].MZ())
    _ = platform.execute([sequence], nshots=None)


@pytest.mark.parametrize(
    "acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION]
)
@pytest.mark.parametrize("batch_size", [None, 3, 5])
def test_dummy_execute_pulse_sequence_unrolling(
    platform: Platform, acquisition, batch_size
):
    nshots = 100
    nsequences = 10
    natives = platform.natives
    sequences = []
    for _ in range(nsequences):
        sequences.append(natives.single_qubit[0].MZ())
    result = platform.execute(sequences, nshots=nshots, acquisition_type=acquisition)
    assert len(next(iter(result.values()))) == nshots
    for r in result.values():
        if acquisition is AcquisitionType.INTEGRATION:
            assert r.shape == (nshots, 2)
        if acquisition is AcquisitionType.DISCRIMINATION:
            assert r.shape == (nshots,)
