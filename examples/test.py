# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from qibolab import Platform
from qibolab.pulses import Pulse, PulseSequence, ReadoutPulse

platform = Platform("tii5q")
qubit = 2

platform.connect()
platform.setup()
platform.start()

ro_pulse = platform.create_MZ_pulse(qubit, start=0)
sequence = PulseSequence()
sequence.add(ro_pulse)
result0 = platform.execute_pulse_sequence(sequence, nshots=10000)

qd_pulse = platform.create_RX_pulse(qubit)
ro_pulse = platform.create_MZ_pulse(qubit, start=qd_pulse.finish)
sequence = PulseSequence()
sequence.add(qd_pulse)
sequence.add(ro_pulse)
result1 = platform.execute_pulse_sequence(sequence, nshots=10000)

platform.stop()
platform.disconnect()

print("Probability 0:", result0.get("probability").get(2))
print("Probability 1:", result1.get("probability").get(2))
print()
print("Shots 0:", np.unique(result0.get("binned_classified").get(2), return_counts=True))
print("Shots 1:", np.unique(result1.get("binned_classified").get(2), return_counts=True))