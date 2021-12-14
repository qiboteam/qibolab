import json
from diagnostics import run_resonator_spectroscopy, \
                        run_qubit_spectroscopy, \
                        run_rabi_pulse_length, \
                        run_rabi_pulse_gain, \
                        run_rabi_pulse_length_and_gain, \
                        run_rabi_pulse_length_and_amplitude, \
                        run_t1, \
                        run_ramsey, \
                        run_spin_echo

import time


if __name__ == "__main__":
    with open("diagnostics_settings.json", "r") as file:
        settings = json.load(file)

    resonator_freq = 7798070000.0
    qubit_freq = 8726500000.0
    pi_pulse_length = 45
    pi_pulse_gain = 0.14

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
    # hardcoded value for t1, ramsey and spin echo
    # from https://github.com/qiboteam/qibolab/blob/singlequbit/diagnostics/tii_single_qubit_diagnosis.ipynb
    print("\nRun t1.\n")
    run_t1(resonator_freq, qubit_freq, pi_pulse_gain,
           pi_pulse_length, **settings["t1"])
    print("\nRun ramsey.\n")
    run_ramsey(resonator_freq, qubit_freq, pi_pulse_gain,
           pi_pulse_length, **settings["ramsey"])
    print("\nRun Spin Echo.\n")
    run_spin_echo(resonator_freq, qubit_freq, pi_pulse_gain,
           pi_pulse_length, **settings["spin_echo"])
    print("\nDiagnostics completed.\n")
