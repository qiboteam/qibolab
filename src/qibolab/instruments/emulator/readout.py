"""Module for simulating qubit readout."""

import numpy as np

from qibolab.pulses import ReadoutPulse
from qibolab.qubits import Qubit


# def lamb_shift(g, delta):
#     """Calculates the lamb shift of the readout resonator from its bare
#     frequency.

#     Args:
#         g (float): Coupling strength between readout resonator and qubit in Hz.
#         delta (float): Detuning between readout resonator and qubit in Hz.

#     Returns:
#         Lambda (float): Lamb shift in Hz
#     """
#     return g * g / delta

# def dispersive_shift(g, delta, alpha):
#     """Calculates the dispersive shift of the readout resonator for the first excited state of the qubit
#     @see https://arxiv.org/abs/2106.06173, equation 35

#     Args:
#         g (float): Coupling strength between readout resonator and qubit in Hz.
#         delta (float): Detuning between readout resonator and qubit in Hz.
#         alpha (float): Qubit anharmonicity in Hz.

#     Returns:
#         chi (float): Dispersive shift in Hz.
#     """
#     return 2* alpha * g * g /delta / delta


def dispersive_shift(g, delta, alpha):
    """Calculates the dispersive shift of the readout resonator for depending on state of the qubit
    @see https://arxiv.org/pdf/1904.06560, equation 146

    Args:
        g (float): Coupling strength between readout resonator and qubit in Hz.
        delta (float): Detuning between readout resonator and qubit in Hz.
        alpha (float): Qubit anharmonicity in Hz.

    Returns:
        chi (float): Dispersive shift in Hz.
    """
    return -1 *g *g /delta *(1/(1+(delta/alpha)))


def s21_function(resonator_frequency, total_Q, coupling_Q):
    """Provides a function that calculates the transmission through the readout
    line at a given frequency.

    Args:
        resonator_frequency (float): Frequency of the resonator.
        total_Q (float): Total Q factor of the resonator.
        coupling_Q (float): Coupling/external Q of the resonator.
    """
    return lambda frequency: 1 - total_Q / np.abs(coupling_Q) / (
        1 + 2j * total_Q * (frequency / resonator_frequency - 1)
    )


class ReadoutSimulator:
    """Module for simulating a readout event based on simulated qubit state."""

    def __init__(
        self, qubit: Qubit, g, noise_model, internal_Q, coupling_Q, sampling_rate=1e9
    ):
        """Initalizer for the readout simulator.

        Args:
            qubit (qibolab.qubits.Qubit): Qibolab qubit object.
            g (float): Coupling strength between readout resonator and qubit in Hz.
            noise_model (function): Time (in)dependent function that models the signal noise.
            internal_Q (float): Internal Q factor of the readout resonator.
            coupling_Q (float): Coupling/external Q factor of the readout resonator.
            sampling_rate (float): Sampling rate of the ADC/digitizer.
        """

        delta = np.abs(qubit.bare_resonator_frequency - qubit.drive_frequency)
        ground_state_frequency = qubit.bare_resonator_frequency - dispersive_shift(g, delta, qubit.anharmonicity)
        excited_state_frequency = qubit.bare_resonator_frequency + dispersive_shift(g, delta, qubit.anharmonicity)
        # excited_state_frequency = ground_state_frequency + dispersive_shift(
        #     g, delta, qubit.anharmonicity
        # )
        self.noise_model = noise_model
        self.sampling_rate = sampling_rate

        total_Q = 1 / (1 / internal_Q + 1 / coupling_Q)
        self.ground_s21 = s21_function(ground_state_frequency, total_Q, coupling_Q)
        self.excited_s21 = s21_function(excited_state_frequency, total_Q, coupling_Q)

    def simulate_ground_state_iq(self, pulse: ReadoutPulse):
        """Simulates the IQ result for a given readout pulse when the qubit is
        in the ground state.

        Args:
            pulse (qibolab.pulses.ReadoutPulse): Qibolab readout pulse.
        """
        s21 = self.ground_s21(pulse.frequency)
        return self.simulate_and_demodulate(s21, pulse)

    def simulate_excited_state_iq(self, pulse: ReadoutPulse):
        """Simulates the IQ result for a given readout pulse when the qubit is
        in the excited state.

        Args:
            pulse (qibolab.pulses.ReadoutPulse): Qibolab readout pulse.
        """
        s21 = self.excited_s21(pulse.frequency)
        return self.simulate_and_demodulate(s21, pulse)

    def simulate_and_demodulate(self, s21: complex, pulse: ReadoutPulse):
        """Simulates the readout pulse for a given S21-parameter and
        demodulates it.

        Args:
            s21 (complex): Complex S21 parameter.
            pulse (qibolab.pulses.ReadoutPulse): Qibolab readout pulse.

        Returns:
            IQ (complex): IQ data for a shot.
        """
        logmag = np.abs(s21)
        phase = np.angle(s21)

        env_I, env_Q = pulse.envelope_waveforms(self.sampling_rate / 1e9)

        start = int(pulse.start * 1e-9 * self.sampling_rate)
        t = np.arange(start, start + len(env_I)) / self.sampling_rate
        reference_signal_I = np.sin(
            2 * np.pi * pulse.frequency * t + pulse.relative_phase
        )
        reference_signal_Q = np.cos(
            2 * np.pi * pulse.frequency * t + pulse.relative_phase
        )

        waveform = (
            logmag
            * pulse.amplitude
            * (
                env_I.data
                * np.cos(pulse.relative_phase + phase)
                * np.sin(2 * np.pi * pulse.frequency * t)
                + env_Q.data
                * np.sin(pulse.relative_phase + phase)
                * np.cos(2 * np.pi * pulse.frequency * t)
                + self.noise_model(t)
            )
        )

        return np.dot(waveform, reference_signal_I) + 1j * np.dot(
            waveform, reference_signal_Q
        )
