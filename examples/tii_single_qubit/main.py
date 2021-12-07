import json
from diagnostics import run_resonator_spectroscopy, \
                        run_qubit_spectroscopy, \
                        run_rabi_pulse_length, \
                        run_rabi_pulse_gain, \
                        run_rabi_pulse_length_and_gain, \
                        run_rabi_pulse_length_and_amplitude


if __name__ == "__main__":
    with open("diagnostics_settings.json", "r") as file:
        settings = json.load(file)

    print("\nRun resonator spectroscopy.\n")
    resonator_freq, _ = run_resonator_spectroscopy(**settings["resonator_spectroscopy"])
    print("\nRun qubit spectroscopy.\n")
    qubit_freq, _ = run_qubit_spectroscopy(**settings["qubit_spectroscopy"])
    print("\nRun Rabi pulse length.\n")
    run_rabi_pulse_length(resonator_freq, qubit_freq)
    print("\nRun Rabi pulse gain.\n")
    run_rabi_pulse_gain(resonator_freq, qubit_freq)
    print("\nRun Rabi pulse length and gain.\n")
    run_rabi_pulse_length_and_gain(resonator_freq, qubit_freq)
    print("\nRun Rabi pulse length and amplitude.\n")
    run_rabi_pulse_length_and_amplitude(resonator_freq, qubit_freq)
    print("\nDiagnostics completed.\n")
