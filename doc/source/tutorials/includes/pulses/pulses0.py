# pulses0.py

from qibolab.pulses import (
    DrivePulse,
    Gaussian,
    PulseSequence,
    ReadoutPulse,
    Rectangular,
)

# Define PulseSequence
sequence = PulseSequence()

# Add some pulses to the pulse sequence
sequence.add(
    DrivePulse(
        start=0,
        frequency=200000000,
        amplitude=0.3,
        duration=60,
        relative_phase=0,
        shape=Gaussian(5),
        qubit=0,
    )
)
sequence.add(
    ReadoutPulse(
        start=70,
        frequency=20000000.0,
        amplitude=0.5,
        duration=3000,
        relative_phase=0,
        shape=Rectangular(),
        qubit=0,
    )
)
