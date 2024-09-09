import numpy as np
import matplotlib.pyplot as plt
from qibolab.qubits import Qubit
from qibolab.instruments.emulator.readout import ReadoutSimulator

bare_resonator_frequency = 5.045e9

qb = Qubit(0, bare_resonator_frequency=bare_resonator_frequency, drive_frequency=3.99e9, anharmonicity=-191.78e6)
readout = ReadoutSimulator(qubit=qb, g=30e6, noise_model=None, internal_Q=2.5e6, coupling_Q=6e4, sampling_rate=1966.08e6)

span = 1e6
center_frequency = bare_resonator_frequency + 0.75e6
freq_sweep = np.linspace(center_frequency - span / 2, center_frequency + span / 2, 1000)
y_gnd = np.abs(readout.ground_s21(freq_sweep))
y_exc = np.abs(readout.excited_s21(freq_sweep))

freq_sweep /= 1e9
plt.plot(freq_sweep, y_gnd, label=r"$|0\rangle$")
plt.plot(freq_sweep, y_exc, label=r"$|1\rangle$")
plt.ylabel("|S21| [arb. units]")
plt.xlabel("Readout Frequency [GHz]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("S21.png", dpi=300)
plt.show()
