from typing import Tuple
import numpy as np
from qibolab import experiment, scheduler
from qibolab.circuit import PulseSequence
from qibolab.pulses import BasicPulse, Rectangular, IQReadoutPulse

def parse_iq(raw_signals, if_frequency):
    final = experiment.readout_params.ADC_sample_size() / experiment.readout_params.ADC_sampling_rate()
    step = 1 / experiment.readout_params.ADC_sampling_rate()
    ADC_time_array = np.arange(0, final, step)[50:]

    i_sig = raw_signals[0]
    q_sig = raw_signals[1]

    cos = np.cos(2 * np.pi * if_frequency * ADC_time_array)
    it = np.sum(i_sig * cos)
    qt = np.sum(q_sig * cos)
    return it, qt


def PulseSpectroscopy(scan_start: float, scan_stop: float, scan_step: float, qubit_channel: int, readout_channels: Tuple[int], readout_if: float,
                      pulse_time: float = 50e6, nshots=10000):
    """Pulse spectroscopy task.
    In this task, we sweep across range of frequencies defined by the scan parameters with a long pulse time.
    At the resonant frequency, the qubit will be driven to the mixed state.

    Returns an array of i, q pairs
    """
    readout_start = 0
    readout_duration = 5e-6
    readout_amplitude = 0.75 / 2 / 5
    phases = (-6.2 / 180 * np.pi, 0.2 / 180 * np.pi)
    readout = IQReadoutPulse(readout_channels, readout_start, readout_duration, readout_amplitude, readout_if, phases)
    duration = 60e-6

    sequence = []
    sweep = np.arange(scan_start, scan_stop, scan_step)
    for freq in sweep:
        drive = BasicPulse(qubit_channel, readout_start - pulse_time, pulse_time, 0.75 / 2, freq, 0, Rectangular())
        sequence.append(PulseSequence([drive, readout], duration=duration))

    job = scheduler.execute_batch_sequence(sequence, nshots)
    result = job.result()
    return [parse_iq(signal, readout_if) for signal in result]

def RabiSpectroscopy(scan_start: float, scan_stop: float, scan_step: float, qubit_channel: int, readout_channels: Tuple[int], readout_if: float,
                     qubit_frequency: float, nshots=10000):
    """Rabi spectroscopy task.
    In this task, we drive the qubit with pulse length depending on the scan parameters.
    From the Rabi oscillation, we can obtain the pi-pulse duration and the Rabi frequency.

    Returns an array of i, q pairs
    """
    readout_start = 0
    readout_duration = 5e-6
    readout_amplitude = 0.75 / 2 / 5
    phases = (-6.2 / 180 * np.pi, 0.2 / 180 * np.pi)
    readout = IQReadoutPulse(readout_channels, readout_start, readout_duration, readout_amplitude, readout_if, phases)

    sequence = []
    sweep = np.arange(scan_start, scan_stop, scan_step)
    for pulse_time in sweep:
        drive = BasicPulse(qubit_channel, readout_start - pulse_time, pulse_time, 0.75 / 2, qubit_frequency, 0, Rectangular())
        sequence.append(PulseSequence([drive, readout]))

    job = scheduler.execute_batch_sequence(sequence, nshots)
    result = job.result()
    return [parse_iq(signal, readout_if) for signal in result]

def RamseyInferometry(scan_start: float, scan_stop: float, scan_step: float, qubit_channel: int, readout_channels: Tuple[int], readout_if: float,
                      qubit_frequency: float, pi_half: float, nshots=10000):
    """Ramsey task.
    In this task, the qubit is driven with two pi-half pulses. The seperation between the two pulses is dependent on the scan parameters.
    The drive frequency is usually detuned from the qubit resonant frequency to determine the frequency.
    The T2 time can be extracted from the decay of the oscillation.
 
    Returns an array of i, q pairs
    """
    readout_start = 0
    readout_duration = 5e-6
    readout_amplitude = 0.75 / 2 / 5
    phases = (-6.2 / 180 * np.pi, 0.2 / 180 * np.pi)
    readout = IQReadoutPulse(readout_channels, readout_start, readout_duration, readout_amplitude, readout_if, phases)

    sequence = []
    sweep = np.arange(scan_start, scan_stop, scan_step)
    second_pulse = BasicPulse(qubit_channel, readout_start - pi_half, pi_half, 0.75 / 2, qubit_frequency, 0, Rectangular())
    for tau in sweep:
        first_pulse = BasicPulse(qubit_channel, readout_start - pi_half - tau - pi_half, pi_half, 0.75 / 2, qubit_frequency, 0, Rectangular())
        sequence.append(PulseSequence([first_pulse, second_pulse, readout]))

    job = scheduler.execute_batch_sequence(sequence, nshots)
    result = job.result()
    return [parse_iq(signal, readout_if) for signal in result]

def T1(scan_start: float, scan_stop: float, scan_step: float, qubit_channel: int, readout_channels: Tuple[int], readout_if: float,
       qubit_frequency: float, pi_pulse: float, nshots=10000):
    """T1 task.
    In this task, the qubit is driven to the excited state by the pi-pulse. Then, a delay tau, which is determined by the scan parameters, is applied before readout.
    The T1 time is determined by the decay of the excited state to the ground state.
 
    Returns an array of i, q pairs
    """
    readout_start = 0
    readout_duration = 5e-6
    readout_amplitude = 0.75 / 2 / 5
    phases = (-6.2 / 180 * np.pi, 0.2 / 180 * np.pi)
    readout = IQReadoutPulse(readout_channels, readout_start, readout_duration, readout_amplitude, readout_if, phases)

    sequence = []
    sweep = np.arange(scan_start, scan_stop, scan_step)
    for tau in sweep:
        if tau >= 0:
            drive = BasicPulse(qubit_channel, readout_start - tau - pi_pulse, pi_pulse, 0.75 / 2, qubit_frequency, 0, Rectangular())
            sequence.append(PulseSequence([drive, readout]))
        else:
            sequence.append(PulseSequence([readout]))

    job = scheduler.execute_batch_sequence(sequence, nshots)
    result = job.result()
    return [parse_iq(signal, readout_if) for signal in result]

def Spinecho(scan_start: float, scan_stop: float, scan_step: float, qubit_channel: int, readout_channels: Tuple[int], readout_if: float,
                      qubit_frequency: float, pi_pulse: float, pi_half: float, nshots=10000):
    """Spin echo (Hahn echo) task.
    In this task, the qubit is driven with a pi-half pulse, a pi-pulse and a pi-half pulse. The seperation tau between the three pulses is dependent on the scan parameters.
    Some spinecho routines fix the first seperation and vary the second seperation. Here, both are varied by the same amount.
    At long tau, the system decays to the mixed state.
 
    Returns an array of i, q pairs
    """
    readout_start = 0
    readout_duration = 5e-6
    readout_amplitude = 0.75 / 2 / 5
    phases = (-6.2 / 180 * np.pi, 0.2 / 180 * np.pi)
    readout = IQReadoutPulse(readout_channels, readout_start, readout_duration, readout_amplitude, readout_if, phases)

    sequence = []
    sweep = np.arange(scan_start, scan_stop, scan_step)
    second_half_pulse = BasicPulse(qubit_channel, readout_start - pi_half, pi_half, 0.75 / 2, qubit_frequency, 0, Rectangular())
    for tau in sweep:
        tau_half = tau / 2
        start =  readout_start - pi_half - tau_half - pi_pulse
        middle_pulse = BasicPulse(qubit_channel, start, pi_pulse, 0.75 / 2, qubit_frequency, 0, Rectangular())
        start = start - tau_half - pi_half
        first_half_pulse = BasicPulse(qubit_channel, start, pi_half, 0.75 / 2, qubit_frequency, 0, Rectangular())
        sequence.append(PulseSequence([first_half_pulse, middle_pulse, second_half_pulse, readout]))

    job = scheduler.execute_batch_sequence(sequence, nshots)
    result = job.result()
    return [parse_iq(signal, readout_if) for signal in result]
