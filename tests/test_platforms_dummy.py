import numpy as np
import pytest

from qibolab.platform import Platform
from qibolab.pulses import PulseSequence


def test_dummy_initialization():
    platform = Platform("dummy")
    platform.reload_settings()
    platform.connect()
    platform.setup()
    platform.start()
    platform.stop()
    platform.disconnect()


def test_dummy_execute_pulse_sequence():
    platform = Platform("dummy")
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_readout_pulse(0, 0))
    result = platform.execute_pulse_sequence(sequence, nshots=100)
