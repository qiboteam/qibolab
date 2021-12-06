import json

from diagnostics import run_resonator_spectroscopy, \
                        run_qubit_spectroscopy

if __name__ == "__main__":
    with open("diagnostics_settings.json", "r") as file:
        settings = json.load(file)

    resonator_freq, _ = run_resonator_spectroscopy(**settings["resonator_spectroscopy"])
    qubit_freq, _ = run_qubit_spectroscopy(**settings["qubit_spectroscopy"])
