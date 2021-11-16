# from tiiq.experiments.abstract import Experiment, Qubit, Resonator
# from tiiq.instruments.qblox import Qblox_QCM, Qblox_QRM
# from tiiq.instruments.rohde_schwarz import SGS100A


# r0 = Resonator(frequency= 7e9, ro_pulse_amplitude= 0.5, ro_pulse_duration= 3e-6)
# q0 = Qubit(frequency= 7e9, pi_pulse_amplitude= 0.5, pi_pulse_duration= 3e-6, T1= 3e-6, T2= 3e-6, resonator= r0)
# qcm = Qblox_QCM()
# qrm = Qblox_QRM()
# sgs100a = SGS100A()

# bfsd_1q3d_rs_qblox = Experiment("bfsd_1q3d_rs_qblox", [r0], [q0], [qcm, qrm, sgs100a])


import numpy as np

from tiiq.instruments.rohde_schwarz import SGS100A
from tiiq.instruments.qblox import Pulsar_QCM
from tiiq.instruments.qblox import Pulsar_QRM

from quantify_core.data.handling import get_datadir, set_datadir
from quantify_core.measurement import MeasurementControl
from quantify_core.measurement.control import Settable, Gettable
import quantify_core.visualization.pyqt_plotmon as pqm
from quantify_core.visualization.instrument_monitor import InstrumentMonitor

QRM_settings = {
		'ref_clock': 'external',
        'gain': 0.5,
        'hardware_avg_en': True,
        'hardware_avg': 1024,
        'sequencer': 0,
        'acq_trigger_mode': 'sequencer',
        'sync_en': True,

        'data_dictionary': 'quantify-data/',
        
        'ro_pulse': {	"freq_if": 20e6,
						"amplitude": 0.5, 
						"length": 6000,
						"offset_i": 0,
						"offset_q": 0,
						"shape": "Block",
						"delay_before": 341,
						"repetition_duration": 200000,
                                      },

        'start_sample': 130,
        'integration_length': 600,
        'sampling_rate': 1e9,
        'mode': 'ssb',
}
QCM_settings = {
	'ref_clock': 'external',
        'gain': 0.5,
        'hardware_avg': 1024,
        'sequencer': 0,
        'sync_en': True,
        'pulse': {	"freq_if": 100e6,
					"amplitude": 0.25, 
					"length": 300,
					"offset_i": 0,
					"offset_q": 0,
					"shape": "Gaussian",
					"delay_before": 1, # cannot be 0
					"repetition_duration": 200000,
                                      },
}
QRM_LO_settings = { "power": 10,
					"frequency":7.79813e9 - QRM_settings['ro_pulse']['freq_if'],
}
QCM_LO_settings = { "power": 12,
					"frequency":8.72e9 + QCM_settings['pulse']['freq_if'],
}

LO_qrm = SGS100A("LO_qrm", '192.168.0.7')
LO_qcm = SGS100A("LO_qcm", '192.168.0.101')
qrm = Pulsar_QRM("qrm", '192.168.0.2')
qcm = Pulsar_QCM("qcm", '192.168.0.3')

LO_qrm.setup(QRM_LO_settings)
LO_qcm.setup(QCM_LO_settings)
qrm.setup(QRM_settings)
qcm.setup(QCM_settings)

MC = MeasurementControl('MC')
plotmon = pqm.PlotMonitor_pyqt('plotmon')
insmon = InstrumentMonitor("Instruments Monitor")

MC.instr_plotmon(plotmon.name)
MC.instrument_monitor(insmon.name)
set_datadir(QRM_settings['data_dictionary'])


def PulseSpectroscopy():
        soft_avg_numnber = 3

        MC.settables(LO_qrm.LO.frequency)
        MC.setpoints(np.arange(-20e6,+20e6,0.5e6) + QRM_LO_settings['frequency'])
        MC.gettables(Gettable(qrm))
        dataset = MC.run('QubitSpec', soft_avg = soft_avg_numnber)
