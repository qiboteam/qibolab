import pathlib
from qibolab.paths import qibolab_folder
from qibolab import Platform
from qibolab.calibration import utils
from qibolab.calibration.calibration import Calibration as Diagnostics

runcard = qibolab_folder / "runcards" / "test.yml"
script_folder = pathlib.Path(__file__).parent
diagnostics_settings = script_folder / "diagnostics.yml"

# Define platform and load specific runcard
platform = Platform("tiiq")
# Connects to lab instruments using the details specified in the calibration settings.
platform.connect()
# Configures instruments using the loaded calibration settings.
platform.setup()

# create a diagnostics/calibration object
ds = Diagnostics(platform, diagnostics_settings)

# Characterisation can be done by changing settings to qibolab/runcards/tiiq.yml and diagnostics.yml
# These scripts do not save the characterisation results on the runcard; to do so use 
#   ds.backup_config_file()
#   resonator_freq, avg_min_voltage, max_ro_voltage, smooth_dataset, dataset = ds.run_resonator_spectroscopy()
#   ds.save_config_parameter('resonator_freq', resonator_freq, 'characterization', 'single_qubit', qubit)

ds.backup_config_file()

save_settings = False

for qubit in platform.qubits:

    ############################
    # Single qubit experiments #
    ############################

    # Cavity spectroscopy
    resonator_freq, avg_min_voltage, max_ro_voltage, smooth_dataset, dataset = ds.run_resonator_spectroscopy(qubit)
    if save_settings:
        ds.save_config_parameter("resonator_freq", int(resonator_freq), 'characterization', 'single_qubit', qubit)
        ds.save_config_parameter("resonator_spectroscopy_avg_ro_voltage", float(avg_min_voltage), 'characterization', 'single_qubit', qubit)
        ds.save_config_parameter("state0_voltage", float(max_ro_voltage), 'characterization', 'single_qubit', qubit)
        lo_qrm_frequency = int(resonator_freq - platform.settings['native_gates']['single_qubit'][qubit]['MZ']['frequency'])
        ds.save_config_parameter("frequency", lo_qrm_frequency, 'instruments', platform.lo_qrm[qubit].name, 'settings')
    
    # Qubit spectroscopy
    qubit_freq, min_ro_voltage, smooth_dataset, dataset = ds.run_qubit_spectroscopy(qubit)
    if save_settings:
        ds.save_config_parameter("qubit_freq", int(qubit_freq), 'characterization', 'single_qubit', qubit)
        RX_pulse_sequence = platform.settings['native_gates']['single_qubit'][qubit]['RX']['pulse_sequence']
        lo_qcm_frequency = int(qubit_freq + RX_pulse_sequence[0]['frequency'])
        ds.save_config_parameter("frequency", lo_qcm_frequency, 'instruments', platform.lo_qcm[qubit].name, 'settings')
        ds.save_config_parameter("qubit_spectroscopy_min_ro_voltage", float(min_ro_voltage), 'characterization', 'single_qubit', qubit)

    # Rabi oscillations
    dataset, pi_pulse_duration, pi_pulse_amplitude, rabi_oscillations_pi_pulse_min_voltage, t1 = ds.run_rabi_pulse_length(qubit)
    if save_settings:
        RX_pulse_sequence[0]['duration'] = int(pi_pulse_duration)
        RX_pulse_sequence[0]['amplitude'] = float(pi_pulse_amplitude)
        ds.save_config_parameter("pulse_sequence", RX_pulse_sequence, 'native_gates', 'single_qubit', qubit, 'RX')
        ds.save_config_parameter("rabi_oscillations_pi_pulse_min_voltage", float(rabi_oscillations_pi_pulse_min_voltage), 'characterization', 'single_qubit', qubit)

    # T1
    t1, smooth_dataset, dataset = ds.run_t1(qubit)
    if save_settings:
        ds.save_config_parameter("T1", float(t2), 'characterization', 'single_qubit', qubit)

    # Ramsey
    t2, delta_frequency, smooth_dataset, dataset = ds.run_ramsey(qubit)
    if save_settings:
        adjusted_qubit_freq = int(platform.characterization['single_qubit'][qubit]['qubit_freq'] + delta_frequency)
        ds.save_config_parameter("qubit_freq", adjusted_qubit_freq, 'characterization', 'single_qubit', qubit)
        ds.save_config_parameter("T2", float(t2), 'characterization', 'single_qubit', qubit)
        RX_pulse_sequence = platform.settings['native_gates']['single_qubit'][qubit]['RX']['pulse_sequence']
        lo_qcm_frequency = int(adjusted_qubit_freq + RX_pulse_sequence[0]['frequency'])
        ds.save_config_parameter("frequency", lo_qcm_frequency, 'instruments', platform.lo_qcm[qubit].name, 'settings')

    # Qubit states
    all_gnd_states, mean_gnd_states, all_exc_states, mean_exc_states = ds.calibrate_qubit_states(qubit)
    if save_settings:
        # print(mean_gnd_states)
        # print(mean_exc_states)
        #TODO: Remove plot qubit states results
        # DEBUG: auto_calibrate_platform - Plot qubit states
        utils.plot_qubit_states(all_gnd_states, all_exc_states)
        ds.save_config_parameter("mean_gnd_states", mean_gnd_states, 'characterization', 'single_qubit', qubit)
        ds.save_config_parameter("mean_exc_states", mean_exc_states, 'characterization', 'single_qubit', qubit)

    # Ramsey auto-detunned frequency
    t2, delta_frequency, smooth_dataset, dataset = ds.run_ramsey_frequency_detuned(qubit)
    if save_settings:
        adjusted_qubit_freq = int(platform.characterization['single_qubit'][qubit]['qubit_freq'] + delta_frequency)
        ds.save_config_parameter("qubit_freq", adjusted_qubit_freq, 'characterization', 'single_qubit', qubit)
        ds.save_config_parameter("T2", float(t2), 'characterization', 'single_qubit', qubit)
        RX_pulse_sequence = platform.settings['native_gates']['single_qubit'][qubit]['RX']['pulse_sequence']
        lo_qcm_frequency = int(adjusted_qubit_freq + RX_pulse_sequence[0]['frequency'])
        ds.save_config_parameter("frequency", lo_qcm_frequency, 'instruments', platform.lo_qcm[qubit].name, 'settings')
    
    # Optimal beta parameter for drag pulses
    beta_optimal = ds.run_drag_pulse_tunning(qubit)
    if save_settings:
        print(beta_optimal)
        # drag_shape = "Drag(" + beta_optimal + ")"
        # ds.save_config_parameter("shape", drag_shape, 'native_gates', 'single_qubit', qubit, 'RX', 'shape')
    
    # Run allXY to test drag shape.
    # COMMENT: run at least once the "run_drag_pulse_tunning" method to save new shape values and check them with allXY results
    results, gateNumber = ds.run_allXY(qubit)
    if save_settings:
        utils.plot_allXY(results, gateNumber)
    
    # Optimal beta parameter for drag pulses
    epsilon = ds.run_flipping(qubit)
    if save_settings:
        print(epsilon) 
        # RX_pulse_amplitude = platform.settings['native_gates']['single_qubit'][qubit]['RX']['amplitude']
        # adjusted_amplitude = RX_pulse_amplitude + epsilon
        # ds.save_config_parameter("amplitude", adjusted_amplitude, 'native_gates', 'single_qubit', qubit, 'RX', 'amplitude')
        # ds.save_config_parameter("amplitude", adjusted_amplitude, 'native_gates', 'single_qubit', qubit, 'MZ', 'amplitude')

    ##########################
    # Multiqubit experiments #
    ##########################

    # ReadOut Matrix
    RO_matrix = ds.run_RO_matrix()
    if save_settings:
        print(RO_matrix)
        utils.saveROMatrix(RO_matrix)

    #####################
    # Other experiments #
    #####################

    # Rabi pulse and gain
    # dataset = ds.run_rabi_pulse_gain(qubit)
    
    