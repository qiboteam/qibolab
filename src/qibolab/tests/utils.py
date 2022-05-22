from qibolab.paths import qibolab_folder
import yaml


def load_runcard(name):
    runcard = qibolab_folder / "runcards" / f"{name}.yml"
    with open(runcard, "r") as file:
        settings = yaml.safe_load(file)
    return settings


def generate_pulse_sequence(readout=True):
    """Generates a dummy pulse sequence to be used for testing pulsar methods."""
    from qibolab.pulses import Pulse, ReadoutPulse, Gaussian, Rectangular
    from qibolab.circuit import PulseSequence
    sequence = PulseSequence()
    sequence.add(Pulse(start=0,
                    frequency=200_000_000,
                    amplitude=0.3,
                    duration=60,
                    phase=0,
                    shape='Gaussian(5)', # Gaussian shape with std = duration / 5
                    channel=1)) 
    sequence.add(Pulse(start=64,
                    frequency=200_000_000,
                    amplitude=0.3,
                    duration=30,
                    phase=0,
                    shape='Gaussian(5)', # Gaussian shape with std = duration / 5
                    channel=1)) 
    if readout:
        sequence.add(ReadoutPulse(start=94,
                          frequency=20_000_000,
                          amplitude=0.9,
                          duration=2000,
                          phase=0,
                          shape='Rectangular()', 
                          channel=11)) 
    return sequence