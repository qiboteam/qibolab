import pathlib
import yaml


def load_runcard(name):
    runcard = pathlib.Path(__file__).parent.parent / "runcards" / f"{name}.yml"
    with open(runcard, "r") as file:
        settings = yaml.safe_load(file)
    return settings


def generate_pulse_sequence():
    """Generates a dummy pulse sequence to be used for testing pulsar methods."""
    from qibolab.pulses import Pulse, ReadoutPulse
    from qibolab.pulse_shapes import Gaussian, Rectangular
    from qibolab.circuit import PulseSequence
    sequence = PulseSequence()
    sequence.add(Pulse(start=0,
                       frequency=200000000.0,
                        amplitude=0.3,
                        duration=60,
                        phase=0,
                        shape=Gaussian(60 / 5)))
    sequence.add(Pulse(start=65,
                       frequency=200000000.0,
                       amplitude=0.8,
                       duration=25,
                       phase=0,
                       shape=Gaussian(25 / 5)))
    sequence.add(ReadoutPulse(start=90,
                              frequency=20000000.0,
                              amplitude=0.5,
                              duration=3000,
                              phase=0,
                              shape=Rectangular()))
    return sequence