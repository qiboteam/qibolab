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
readout = ReadoutSimulator(qubit=qb, g=15e6, noise_model=AWGN, internal_Q=2.5e6, coupling_Q=6e4, sampling_rate=1966.08e6)
ro_pulse = ReadoutPulse(start=0, duration=1000, amplitude=READOUT_AMPLITUDE, frequency=5.0450000000001e9, shape="Rectangular()", relative_phase=0)


#demonstrates effect of state dependent dispersive shift on amplitude of reflected microwave from resonator; 
span = 1e6
center_frequency = bare_resonator_frequency 
freq_sweep = np.linspace(center_frequency - span / 2, center_frequency + span / 2, 1000)
print(readout.ground_s21(freq_sweep))
y_gnd = np.abs(readout.ground_s21(freq_sweep))
y_exc = np.abs(readout.excited_s21(freq_sweep))

# freq_sweep /= 1e9
# plt.plot(freq_sweep, y_gnd, label=r"$|0\rangle$")
# plt.plot(freq_sweep, y_exc, label=r"$|1\rangle$")
# plt.ylabel("|S21| [arb. units]")
# plt.xlabel("Readout Frequency [GHz]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# # plt.savefig("S21.png", dpi=300)
# plt.show()

# freq_sweep *= 1e9

# #demonstrates effect of state dependent dispersive shift on phase of reflected microwave from resonator; (for domain of [-pi/2,pi/2])
# y_gnd1 = np.angle(readout.ground_s21(freq_sweep))
# y_exc1 = np.angle(readout.excited_s21(freq_sweep))
# freq_sweep /= 1e9
# plt.plot(freq_sweep, y_gnd1, label=r"$|0\rangle$")
# plt.plot(freq_sweep, y_exc1, label=r"$|1\rangle$")
# plt.ylabel(r"$\theta$ phase [rad]")
# plt.xlabel("Readout Frequency [GHz]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# # plt.savefig("phase.png", dpi=300)
# plt.show()

#demonstrates the separation of V_I/V_Q data of reflected microwave on V_I/V_Q plane
rgnd = [readout.simulate_ground_state_iq(ro_pulse) for k in range(nshots)]
rexc = [readout.simulate_excited_state_iq(ro_pulse) for k in range(nshots)]
plt.scatter(np.real(rgnd), np.imag(rgnd), label=r"$|0\rangle$")
plt.scatter(np.real(rexc), np.imag(rexc), label=r"$|1\rangle$")
plt.xlabel("V_I")
plt.ylabel("V_Q")
plt.legend()
plt.tight_layout()
# plt.savefig("IQ.png", dpi=300)
plt.show()





