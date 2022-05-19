from qibolab import Platform
import diagnostics
from qibolab.calibration import utils

# Create a platform; connect and configure it
platform = Platform('multiqubit')
platform.connect()
platform.setup()

# create a diagnostics object
ds = diagnostics.Diagnostics(platform)


utils.backup_config_file(platform)

# Characterisation can be done by changing settings to qibolab/runcards/tiiq.yml and diagnostics.yml

resonator_freq, avg_min_voltage, max_ro_voltage, smooth_dataset, dataset = ds.run_resonator_spectroscopy()

qubit_freq, min_ro_voltage, smooth_dataset, dataset = ds.run_qubit_spectroscopy()

dataset, pi_pulse_duration, pi_pulse_amplitude, pi_pulse_gain, rabi_oscillations_pi_pulse_min_voltage, t1 = ds.run_rabi_pulse_length()

t1, smooth_dataset, dataset = ds.run_t1()

t2, smooth_dataset, dataset = ds.run_ramsey()

