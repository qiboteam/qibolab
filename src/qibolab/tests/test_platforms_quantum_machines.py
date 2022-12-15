import json
import pathlib

from qibolab import Platform
from qibolab.pulses import PulseSequence

REGRESSION_FOLDER = pathlib.Path(__file__).with_name("regressions")


def assert_regression_fixture(results, filename, rtol=1e-7, atol=1e-12):
    """Check result match data inside filename."""

    filename = REGRESSION_FOLDER / filename
    try:
        with open(filename) as file:
            results_fixture = json.load(file)
    except:  # pragma: no cover
        # case not tested in GitHub workflows because files exist
        with open(filename, "w") as file:
            json.dump(results, file)
        with open(filename) as file:
            results_fixture = json.load(file)
    results_serial = json.loads(json.dumps(results))
    assert results_serial == results_fixture


def test_initial_config():
    platform = Platform("qm")
    assert_regression_fixture(platform.config, "qm_initial_config.json")


def test_register_drive_pulse():
    from qibolab.pulses import Gaussian, Pulse

    pulse = Pulse(0, 20, 0.2, 20000000, 0.0, Gaussian(5), channel=0, qubit=0)
    platform = Platform("qm")
    platform.register_pulse(pulse)

    print(platform.config["waveforms"])
    assert platform.config["waveforms"] == {
        "Envelope_Waveform_I(num_samples = 20, amplitude = 0.2, shape = Gaussian(5))": {
            "type": "arbitrary",
            "samples": pulse.envelope_waveform_i.data.tolist(),
        },
        "Envelope_Waveform_Q(num_samples = 20, amplitude = 0.2, shape = Gaussian(5))": {
            "type": "arbitrary",
            "samples": pulse.envelope_waveform_q.data.tolist(),
        },
    }
    assert platform.config["pulses"] == {
        pulse.serial: {
            "operation": "control",
            "length": 20,
            "waveforms": {
                "I": "Envelope_Waveform_I(num_samples = 20, amplitude = 0.2, shape = Gaussian(5))",
                "Q": "Envelope_Waveform_Q(num_samples = 20, amplitude = 0.2, shape = Gaussian(5))",
            },
        }
    }
    assert platform.config["elements"]["drive0"]["operations"] == {pulse.serial: pulse.serial}
    assert platform.config["mixers"] == {
        "mixer_drive0": [
            {"intermediate_frequency": 20000000, "lo_frequency": None, "correction": [1.0, 0.0, 0.0, 1.0]}
        ],
        "mixer_readout0": [{"intermediate_frequency": 0, "lo_frequency": None, "correction": [1.0, 0.0, 0.0, 1.0]}],
    }


def test_register_readout_pulse():
    from qibolab.pulses import ReadoutPulse, Rectangular

    pulse = ReadoutPulse(0, 20, 0.2, 20000000, 0.0, Rectangular(), channel=0, qubit=1)
    platform = Platform("qm")
    platform.register_pulse(pulse)

    assert platform.config["waveforms"] == {"constant_wf0.2": {"type": "constant", "sample": 0.2}}
    assert platform.config["pulses"] == {
        pulse.serial: {
            "operation": "measurement",
            "length": 20,
            "waveforms": {"I": "constant_wf0.2", "Q": "constant_wf0.2"},
            "integration_weights": {
                "cos": "cosine_weights1",
                "sin": "sine_weights1",
                "minus_sin": "minus_sine_weights1",
            },
            "digital_marker": "ON",
        }
    }
    assert platform.config["elements"]["readout1"]["operations"] == {pulse.serial: pulse.serial}
    assert platform.config["integration_weights"] == {
        "cosine_weights1": {"cosine": [(1.0, 20)], "sine": [(-0.0, 20)]},
        "minus_sine_weights1": {"cosine": [(-0.0, 20)], "sine": [(-1.0, 20)]},
        "sine_weights1": {"cosine": [(0.0, 20)], "sine": [(1.0, 20)]},
    }
    assert platform.config["mixers"] == {
        "mixer_drive1": [{"intermediate_frequency": 0, "lo_frequency": None, "correction": [1.0, 0.0, 0.0, 1.0]}],
        "mixer_readout1": [
            {"intermediate_frequency": 20000000, "lo_frequency": None, "correction": [1.0, 0.0, 0.0, 1.0]}
        ],
    }


def test_register_flux_pulse():
    from qibolab.pulses import FluxPulse, Rectangular

    pulse = FluxPulse(0, 20, 0.2, 0.0, Rectangular(), channel=0, qubit=2)
    platform = Platform("qm")
    platform.register_pulse(pulse)

    assert platform.config["waveforms"] == {"constant_wf0.2": {"type": "constant", "sample": 0.2}}
    assert platform.config["pulses"] == {
        pulse.serial: {
            "operation": "control",
            "length": 20,
            "waveforms": {"single": "constant_wf0.2"},
        }
    }
    assert platform.config["elements"]["flux2"]["operations"] == {pulse.serial: pulse.serial}


# TODO: Test different configurations of pulse sequence executions
# TODO: Test cases where pulses are played on different connectors
# this will require to update the ``SimulationConfig`` in the platform


def test_pulse_sequence_execution_simulated_waveforms(qmsim_address):
    platform = Platform("qm")
    platform.connect(qmsim_address)
    platform.setup()

    qd_pulse = platform.create_RX_pulse(1, start=0)
    qd_pulse2 = platform.create_RX_pulse(1, start=qd_pulse.duration)
    ro_pulse = platform.create_MZ_pulse(1, start=2 * qd_pulse.duration)
    sequence = PulseSequence()
    sequence.add(qd_pulse)
    sequence.add(qd_pulse2)
    sequence.add(ro_pulse)

    result = platform.execute_pulse_sequence(sequence, nshots=1, simulation_duration=1000)
    samples = result.get_simulated_samples()
    samples_sparse = {
        port: [(i, v) for i, v in enumerate(values) if v != 0] for port, values in samples.con1.analog.items()
    }
    assert_regression_fixture(samples_sparse, "qm_sequence_execution.json")
