import numpy as np
import matplotlib.pyplot as plt
from qibolab.qubits import Qubit
from qibolab.instruments.emulator.readout import ReadoutSimulator
from qibolab.pulses import ReadoutPulse

bare_resonator_frequency = 5.045e9
nshots = 100

SNR = 50 # dB
READOUT_AMPLITUDE = 1
NOISE_AMP = np.power(10, -SNR / 20)
AWGN = lambda t: np.random.normal(loc=0, scale=NOISE_AMP, size=len(t))

qb = Qubit(0, bare_resonator_frequency=bare_resonator_frequency, drive_frequency=3.99e9, anharmonicity=-191.78e6)
readout = ReadoutSimulator(qubit=qb, g=1000e6, noise_model=AWGN, internal_Q=2.5e6, coupling_Q=6e4, sampling_rate=1966.08e6)
ro_pulse = ReadoutPulse(start=0, duration=1000, amplitude=READOUT_AMPLITUDE, frequency=5.04585e9, shape="Rectangular()", relative_phase=0)

print(qb.bare_resonator_frequency-qb.drive_frequency)
span = 1000e6
center_frequency = bare_resonator_frequency + 0.75e6
freq_sweep = np.linspace(center_frequency - span / 2, center_frequency + span / 2, 1000)
y_gnd = np.abs(readout.ground_s21(freq_sweep))
y_exc = np.abs(readout.excited_s21(freq_sweep))

freq_sweep /= 1e9
plt.plot(freq_sweep, y_gnd, label=r"$|0\rangle$")
plt.plot(freq_sweep, y_exc, label=r"$|1\rangle$")
plt.ylabel(r"$\theta$ phase [rad]")
plt.xlabel("Readout Frequency [GHz]")
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig("S21.png", dpi=300)
plt.show()


# rgnd = [readout.simulate_ground_state_iq(ro_pulse) for k in range(nshots)]
# rexc = [readout.simulate_excited_state_iq(ro_pulse) for k in range(nshots)]
# plt.scatter(np.real(rgnd), np.imag(rgnd), label=r"$|0\rangle$")
# plt.scatter(np.real(rexc), np.imag(rexc), label=r"$|1\rangle$")
# plt.xlabel("I")
# plt.ylabel("Q")
# plt.legend()
# plt.tight_layout()
# #plt.savefig("IQ.png", dpi=300)
# plt.show()