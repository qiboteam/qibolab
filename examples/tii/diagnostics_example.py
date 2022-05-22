from qibolab import Platform
import diagnostics
from qibolab.calibration import utils


# Define platform and load specific runcard
platform = Platform("multiqubit")
# Connects to lab instruments using the details specified in the calibration settings.
platform.connect()
# Configures instruments using the loaded calibration settings.
platform.setup()

# create a diagnostics object
ds = diagnostics.Diagnostics(platform)

# Characterisation can be done by changing settings to qibolab/runcards/tiiq.yml and diagnostics.yml
# These scripts do not save the characterisation results on the runcard; to do so use 
#   from qibolab.calibration import utils
#   utils.backup_config_file(platform)
#   resonator_freq, avg_min_voltage, max_ro_voltage, smooth_dataset, dataset = ds.run_resonator_spectroscopy()
#   utils.save_config_parameter("settings", "", "resonator_freq", float(resonator_freq))

resonator_freq, avg_min_voltage, max_ro_voltage, smooth_dataset, dataset = ds.run_resonator_spectroscopy()

qubit_freq, min_ro_voltage, smooth_dataset, dataset = ds.run_qubit_spectroscopy()

dataset, pi_pulse_duration, pi_pulse_amplitude, pi_pulse_gain, rabi_oscillations_pi_pulse_min_voltage, t1 = ds.run_rabi_pulse_length()

t1, smooth_dataset, dataset = ds.run_t1()

t2, smooth_dataset, dataset = ds.run_ramsey()

