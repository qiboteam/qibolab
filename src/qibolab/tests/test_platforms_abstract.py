# -*- coding: utf-8 -*-
import numpy as np
import pytest
from qibo import gates

from qibolab.platform import Platform
from qibolab.pulses import PulseSequence


def test_u3_to_sequence():
    platform = Platform("tii5q")
    gate = gates.U3(0, theta=0.1, phi=0.2, lam=0.3)
    sequence = PulseSequence()
    platform.to_sequence(sequence, gate)
    assert len(sequence) == 2


def test_measurement():
    platform = Platform("tii5q")
    gate = gates.M(0)
    sequence = PulseSequence()
    platform.to_sequence(sequence, gate)
    assert len(sequence) == 1
    assert len(sequence.qd_pulses) == 0
    assert len(sequence.qf_pulses) == 0
    assert len(sequence.ro_pulses) == 1


def test_rz_to_sequence():
    platform = Platform("tii5q")
    sequence = PulseSequence()
    platform.to_sequence(sequence, gates.RZ(0, theta=0.2))
    platform.to_sequence(sequence, gates.Z(0))
    assert len(sequence) == 0
    assert sequence.phase == 0.2 + np.pi


@pytest.mark.parametrize("platform_name", ["tii1q", "tii5q"])  # , 'icarusq'])
def test_pulse_sequence_add_u3(platform_name):
    platform = Platform(platform_name)
    seq = PulseSequence()
    platform.to_sequence(seq, gates.U3(0, 0.1, 0.2, 0.3))
    assert len(seq.pulses) == 2
    assert len(seq.qd_pulses) == 2

    RX90_pulse1 = platform.RX90_pulse(0, start=0, phase=0.3)
    RX90_pulse2 = platform.RX90_pulse(0, start=(RX90_pulse1.start + RX90_pulse1.duration), phase=0.4 - np.pi)

    np.testing.assert_allclose(seq.time, RX90_pulse1.duration + RX90_pulse2.duration)
    np.testing.assert_allclose(seq.phase, 0.6)
    assert seq.serial == f"{RX90_pulse1.serial}, {RX90_pulse2.serial}"


@pytest.mark.parametrize("platform_name", ["tii1q", "tii5q"])  # , 'icarusq'])
def test_pulse_sequence_add_two_u3(platform_name):
    platform = Platform(platform_name)
    seq = PulseSequence()
    platform.to_sequence(seq, gates.U3(0, 0.1, 0.2, 0.3))
    platform.to_sequence(seq, gates.U3(0, 0.4, 0.6, 0.5))
    assert len(seq.pulses) == 4
    assert len(seq.qd_pulses) == 4

    RX90_pulse = platform.RX90_pulse(0)
    np.testing.assert_allclose(seq.phase, 0.6 + 1.5)
    np.testing.assert_allclose(seq.time, 2 * 2 * RX90_pulse.duration)

    RX90_pulse1 = platform.RX90_pulse(0, start=0, phase=0.3)
    RX90_pulse2 = platform.RX90_pulse(0, start=(RX90_pulse1.start + RX90_pulse1.duration), phase=0.4 - np.pi)
    RX90_pulse3 = platform.RX90_pulse(0, start=(RX90_pulse2.start + RX90_pulse2.duration), phase=1.1)
    RX90_pulse4 = platform.RX90_pulse(0, start=(RX90_pulse3.start + RX90_pulse3.duration), phase=1.5 - np.pi)

    assert seq.serial == f"{RX90_pulse1.serial}, {RX90_pulse2.serial}, {RX90_pulse3.serial}, {RX90_pulse4.serial}"


@pytest.mark.parametrize("platform_name", ["tii1q", "tii5q"])  # , 'icarusq'])
def test_pulse_sequence_add_measurement(platform_name):
    platform = Platform(platform_name)
    seq = PulseSequence()
    platform.to_sequence(seq, gates.U3(0, 0.1, 0.2, 0.3))
    platform.to_sequence(seq, gates.M(0))
    assert len(seq.pulses) == 3
    assert len(seq.qd_pulses) == 2
    assert len(seq.ro_pulses) == 1

    np.testing.assert_allclose(seq.phase, 0.6)

    RX90_pulse1 = platform.RX90_pulse(0, start=0, phase=0.3)
    RX90_pulse2 = platform.RX90_pulse(0, start=RX90_pulse1.duration, phase=0.4 - np.pi)
    MZ_pulse = platform.MZ_pulse(0, start=(RX90_pulse2.start + RX90_pulse2.duration), phase=0.6)
    assert seq.serial == f"{RX90_pulse1.serial}, {RX90_pulse2.serial}, {MZ_pulse.serial}"
