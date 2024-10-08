import os

import matplotlib.pyplot as plt
import numpy as np

import qibolab
from qibolab.pulses import PulseSequence

os.environ["QIBOLAB_PLATFORMS"] = (
    r"C:\Users\LexNg\OneDrive - Nanyang Technological University\YEAR 4 - FYP\qibolab\qibolab\tests\emulators"
)
# os.environ["QIBOLAB_PLATFORMS"] = r"C:\Users\pault\Documents\repos\qibolab\tests\emulators"
platform = qibolab.create_platform("ibmfakebelem_q01")

pi_pulse = platform.create_RX_pulse(0)
readout_pulse = platform.create_qubit_readout_pulse(qubit=0, start=pi_pulse.finish)
# #for two readout pulses
# pi_pulse2 = platform.create_RX90_pulse(0, start = readout_pulse.finish)
# readout_pulse2 = platform.create_qubit_readout_pulse(qubit=0, start=pi_pulse2.finish)

SNR = 30  # dB
NOISE_AMP = np.power(10, -SNR / 20)
AWGN = lambda t: np.random.normal(loc=0, scale=NOISE_AMP, size=len(t)) * 3e4

ps = PulseSequence(*[readout_pulse])
opts = qibolab.ExecutionParameters(
    nshots=100,
    relaxation_time=100e3,
    acquisition_type=qibolab.AcquisitionType.DISCRIMINATION,
    averaging_mode=qibolab.AveragingMode.SINGLESHOT,
    readout_simulator_config=dict(
        g=10e6,
        noise_model=AWGN,
        internal_Q=2.5e6,
        coupling_Q=6e4,
        sampling_rate=1966.08e6,
    ),
)

gnd = platform.execute_pulse_sequence(ps, opts)
print(gnd)
print(gnd[0].probability())

ps.add(pi_pulse)
excited = platform.execute_pulse_sequence(ps, opts)
print(excited[0].probability())

i_ground = gnd["demodulation"][0]["i"]
q_ground = gnd["demodulation"][0]["q"]

i_excited = excited["demodulation"][0]["i"]
q_excited = excited["demodulation"][0]["q"]

# evenly distributed as measurement of ground state(excited state) results in
# excited state(ground state) due to noise and statistical fluctuations in measurement
plt.scatter(i_ground, q_ground, label=r"$|0\rangle$")
plt.scatter(i_excited, q_excited, label=r"$|1\rangle$")

plt.xlabel(r"$V_I$ [a.u.]")
plt.ylabel(r"$V_Q$ [a.u.]")

# #when scale in noise_model = 0
# plt.xlim([0,0.8])

plt.legend()
plt.tight_layout()
# plt.savefig("IQ_pulse_simulator.png", dpi=300)
plt.show()
