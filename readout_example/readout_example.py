import matplotlib.pyplot as plt
import numpy as np

from qibolab.instruments.emulator.readout import ReadoutSimulator
from qibolab.pulses import ReadoutPulse
from qibolab.qubits import Qubit

bare_resonator_frequency = 5.045e9
nshots = 100

SNR = 40  # dB
READOUT_AMPLITUDE = 1
NOISE_AMP = np.power(10, -SNR / 20)
AWGN = lambda t: np.random.normal(loc=0, scale=NOISE_AMP, size=len(t)) * 4.5e1

qb = Qubit(
    0,
    bare_resonator_frequency=bare_resonator_frequency,
    drive_frequency=3.99e9,
    anharmonicity=-263e6,
)
readout = ReadoutSimulator(
    qubit=qb,
    g=10e6,
    noise_model=AWGN,
    internal_Q=2.5e6,
    coupling_Q=6e4,
    sampling_rate=1966.08e6,
)


# demonstrates effect of state dependent dispersive shift on amplitude of reflected microwave from resonator;
# we first prepare a centre frequency for frequency sweeping, which may be modified during analysis, to search for the dispersive shift
# note that dispersive shift and lamb shift depends on detuning(i.e.: drive_frequency - bare_resonator_frequency)
# lamb shifted frequncy would then be shifted dispersively depending on ground_state or excited state of qubit
# fitting of |S21| is discussed here: https://github.com/qiboteam/qibocal/pull/917
span = 1e6
center_frequency = bare_resonator_frequency
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
# plt.savefig("S21.png", dpi=300)
plt.show()

freq_sweep *= 1e9


# demonstrates effect of state dependent dispersive shift on phase of reflected microwave from resonator; (for codomain of [-pi/2,pi/2])
# note that we can always shift the phase/angle by pi, as a result of arctan(), as there are two angles resulting in the same tan() value
# this means that we can shift the positive angles downwards by subtracting them with pi, resulting in codomain of [0,-2pi]
# for a clearer picture (using codomain of [0,-2pi] @see https://arxiv.org/pdf/1904.06560), we can see that
# the phase response of resonator shall be maximally separated when resonator is probed just in-between two qubit-state dependent resonance frequencies.
y_gnd1 = np.angle(readout.ground_s21(freq_sweep))
y_exc1 = np.angle(readout.excited_s21(freq_sweep))
freq_sweep /= 1e9
plt.plot(freq_sweep, y_gnd1, label=r"$|0\rangle$")
plt.plot(freq_sweep, y_exc1, label=r"$|1\rangle$")
plt.ylabel(r"$\theta$ phase [rad]")
plt.xlabel("Readout Frequency [GHz]")
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig("phase.png", dpi=300)
plt.show()


# demonstrates the separation of V_I/V_Q data of reflected microwave on V_I/V_Q plane
# we prepare a lambshifted readout pulse frequency according to the first and second demonstration,
# so that we can inspect the phase response of resonator (being plotted on V_I/V_Q plane),
# which shall be maximally separated when resonator is probed just in-between two qubit-state dependent resonance frequencies.
# in other words, the data probed from resonator for particular qubit states should be well separated as demonstrated previously
ro_frequency = 5.0450e9 + readout.lambshift
ro_pulse = ReadoutPulse(
    start=0,
    duration=1000,
    amplitude=READOUT_AMPLITUDE,
    frequency=ro_frequency,
    shape="Rectangular()",
    relative_phase=0,
)

rgnd = [readout.simulate_ground_state_iq(ro_pulse) for k in range(nshots)]
rexc = [readout.simulate_excited_state_iq(ro_pulse) for k in range(nshots)]
plt.scatter(np.real(rgnd), np.imag(rgnd), label=r"$|0\rangle$")
plt.scatter(np.real(rexc), np.imag(rexc), label=r"$|1\rangle$")
# when we set NOISE_AMP to zero, using the follow axes limits allow us to see the maximally separated data
# plt.xlim([0,1])
# plt.ylim([-1,1])
plt.xlabel(r"$V_I$")
plt.ylabel(r"$V_Q$")
plt.legend()
plt.tight_layout()
plt.savefig("IQ.png", dpi=300)
plt.show()
