from tiiq.experiments.abstract import Experiment, Qubit, Resonator
from tiiq.instruments.qblox import Qblox_QCM, Qblox_QRM
from tiiq.instruments.rohde_schwarz import SGS100A


r0 = Resonator(frequency= 7e9, ro_pulse_amplitude= 0.5, ro_pulse_duration= 3e-6)
q0 = Qubit(frequency= 7e9, pi_pulse_amplitude= 0.5, pi_pulse_duration= 3e-6, T1= 3e-6, T2= 3e-6, resonator= r0)
qcm = Qblox_QCM()
qrm = Qblox_QRM()
sgs100a = SGS100A()

bfsd_1q3d_rs_qblox = Experiment("bfsd_1q3d_rs_qblox", [r0], [q0], [qcm, qrm, sgs100a])


