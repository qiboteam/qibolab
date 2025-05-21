"""Testing basic pulse sequences with emulator platforms."""

import numpy as np
import pytest

from qibolab import Platform, Pulse, PulseSequence, Rectangular, VirtualZ


def test_ground_state(platform: Platform):
    """Test the ground state of a qubit with emulator."""
    q0 = platform.natives.single_qubit[0]
    seq = q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=1e-2) == 0


def test_superposition_state(platform: Platform):
    """Test superposition state of a qubit with emulator."""
    q0 = platform.natives.single_qubit[0]
    seq = q0.RX90() | q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=5e-2) == 0.5


def test_excited_state(platform: Platform):
    """Test excited state of a qubit with emulator."""
    q0 = platform.natives.single_qubit[0]
    seq = q0.RX() | q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=5e-2) == 1


def test_second_excited_state(platform: Platform):
    """Test second excited state of a qubit with emulator."""
    q0 = platform.natives.single_qubit[0]
    if q0.RX12 is None:
        pytest.skip(f"Skipping due to missing RX12 for platform {platform}.")
    seq = q0.RX() | q0.RX12() | q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=1e-1) == 2


def test_virtualz_sequence(platform: Platform):
    """Test virtual Z sequence of a qubit with emulator."""
    q0 = platform.natives.single_qubit[0]
    ch = platform.qubits[0].drive
    seq = (
        q0.RX90()
        + PulseSequence([(ch, VirtualZ(phase=-np.pi / 2))])
        + q0.R(theta=np.pi / 2, phi=np.pi / 2)
    )
    seq |= q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=5e-2) == 1


def test_detuning_flux_pulse(platform: Platform):
    """Test detuning caused by flux pulse."""
    q0 = platform.natives.single_qubit[0]
    flux_channel = platform.qubits[0].flux
    if flux_channel not in platform.channels:
        pytest.skip(f"Skipping due to missing flux channel for platform {platform}.")
    # move the qubit away while applying RX pulse
    seq = PulseSequence(
        [
            (flux_channel, Pulse(duration=40, amplitude=0.8, envelope=Rectangular())),
        ]
    )
    seq += q0.RX()
    seq |= q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    assert pytest.approx(res.mean(), abs=5e-2) != 1


def test_detuning_static_bias(platform: Platform):
    """Test detuning caused by static bias."""
    q0 = platform.natives.single_qubit[0]
    flux_channel = platform.qubits[0].flux
    if flux_channel not in platform.channels:
        pytest.skip(f"Skipping due to missing flux channel for platform {platform}.")
    seq = q0.RX() | q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    updates = [{flux_channel: {"offset": 0.8}}]
    res = platform.execute([seq], nshots=1e4, updates=updates)[acq_handle]
    assert pytest.approx(res.mean(), abs=5e-2) != 1


def test_detuning_second_excited_state(platform: Platform):
    """Test detuning second excited state of a qubit with emulator."""
    q0 = platform.natives.single_qubit[0]
    flux_channel = platform.qubits[0].flux
    if q0.RX12 is None:
        pytest.skip(f"Skipping due to missing RX12 for platform {platform}.")
    if flux_channel not in platform.channels:
        pytest.skip(f"Skipping due to missing flux channel for platform {platform}.")
    seq = q0.RX()
    seq |= [
        (flux_channel, Pulse(duration=40, amplitude=0.8, envelope=Rectangular()))
    ] + q0.RX12()
    seq |= q0.MZ()
    acq_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)[acq_handle]
    # since we apply a flux pulse the qubit stays at 1
    assert pytest.approx(res.mean(), abs=1e-1) == 1


@pytest.mark.parametrize("setup", ["Id", "X"])
def test_cnot_sequence(platform: Platform, setup: str):
    """Test CNOT sequence with emulator."""
    if platform.nqubits < 2:
        pytest.skip(f"Plaform {platform} requires at least two qubits.")
    if platform.natives.two_qubit[0, 1].CNOT is None:
        pytest.skip(f"Skipping due to missing CNOT for platform {platform}.")
    q0 = platform.natives.single_qubit[0]
    q1 = platform.natives.single_qubit[1]
    pair = platform.natives.two_qubit[0, 1]
    seq = PulseSequence()
    if setup == "X":
        seq += q0.RX()

    seq = pair.CNOT()
    seq |= q0.MZ() + q1.MZ()
    target_handle = list(seq.channel(platform.qubits[1].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)
    assert (
        pytest.approx(res[target_handle].mean(), abs=2e-1) == 0 if setup == "Id" else 1
    )


def test_cz_sequence(
    platform: Platform,
):
    """Test CZ sequence with emulator."""
    if platform.nqubits < 2:
        pytest.skip(f"Plaform {platform} requires at least two qubits.")
    if platform.natives.two_qubit[0, 1].CZ is None:
        pytest.skip(f"Skipping due to missing CNOT for platform {platform}.")
    q0 = platform.natives.single_qubit[0]
    q1 = platform.natives.single_qubit[1]
    pair = platform.natives.two_qubit[0, 1]

    seq = PulseSequence()
    seq += q0.R(theta=np.pi / 2, phi=np.pi / 2)
    seq += q1.R(theta=np.pi / 2, phi=np.pi / 2)
    seq |= pair.CZ()
    seq |= q0.RX() + q1.R(theta=np.pi / 2, phi=np.pi / 2)
    seq |= q0.MZ() + q1.MZ()
    control_handle = list(seq.channel(platform.qubits[0].acquisition))[-1].id
    target_handle = list(seq.channel(platform.qubits[1].acquisition))[-1].id
    res = platform.execute([seq], nshots=1e4)
    assert pytest.approx(res[target_handle].mean(), abs=2e-1) == 0.5
    assert pytest.approx(res[control_handle].mean(), abs=2e-1) == 0.5
