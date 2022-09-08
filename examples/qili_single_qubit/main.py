# -*- coding: utf-8 -*-
import time

import yaml
from diagnostics import (
    run_qubit_spectroscopy,
    run_rabi_pulse_gain,
    run_rabi_pulse_length,
    run_rabi_pulse_length_and_amplitude,
    run_rabi_pulse_length_and_gain,
    run_ramsey,
    run_resonator_spectroscopy,
    run_spin_echo,
    run_t1,
)

if __name__ == "__main__":
    with open("settings.yaml", "r") as file:
        settings = yaml.safe_load(file)

    resonator_freq = 7798070000.0
    qubit_freq = 8726500000.0
    pi_pulse_length = 45
    pi_pulse_gain = 0.14
    pi_pulse_amplitude = 0.9

    print("\nRun resonator spectroscopy.\n")
    resonator_freq, _ = run_resonator_spectroscopy(**settings["resonator_spectroscopy"])
    print("\nRun qubit spectroscopy.\n")
    qubit_freq, _ = run_qubit_spectroscopy(resonator_freq, **settings["qubit_spectroscopy"])
    print("\nRun Rabi pulse length.\n")
    run_rabi_pulse_length(resonator_freq, qubit_freq)
    print("\nRun Rabi pulse gain.\n")
    run_rabi_pulse_gain(resonator_freq, qubit_freq)
    print("\nRun Rabi pulse length and gain.\n")
    run_rabi_pulse_length_and_gain(resonator_freq, qubit_freq)
    print("\nRun Rabi pulse length and amplitude.\n")
    run_rabi_pulse_length_and_amplitude(resonator_freq, qubit_freq)
    print("\nRun t1.\n")
    run_t1(resonator_freq, qubit_freq, pi_pulse_gain, pi_pulse_length, **settings["t1"])
    print("\nRun ramsey.\n")
    run_ramsey(resonator_freq, qubit_freq, pi_pulse_gain, pi_pulse_length, pi_pulse_amplitude, **settings["ramsey"])
    print("\nRun Spin Echo.\n")
    run_spin_echo(
        resonator_freq, qubit_freq, pi_pulse_gain, pi_pulse_length, pi_pulse_amplitude, **settings["spin_echo"]
    )
    print("\nDiagnostics completed.\n")

    time.sleep(360)
