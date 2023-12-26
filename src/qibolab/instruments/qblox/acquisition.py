from dataclasses import dataclass
from typing import List, Optional

import numpy as np

SAMPLING_RATE = 1


def demodulate(input_i, input_q, frequency):
    """Demodulates and integrates the acquired pulse."""
    # DOWN Conversion
    # qblox does not remove the offsets in hardware
    modulated_i = input_i - np.mean(input_i)
    modulated_q = input_q - np.mean(input_q)

    num_samples = modulated_i.shape[0]
    time = np.arange(num_samples)
    phase = 2 * np.pi * frequency * time / SAMPLING_RATE
    cosalpha = np.cos(phase)
    sinalpha = np.sin(phase)
    demod_matrix = np.sqrt(2) * np.array([[cosalpha, sinalpha], [-sinalpha, cosalpha]])
    result = np.einsum("ijt,jt->i", demod_matrix, np.stack([modulated_i, modulated_q]))
    return np.sqrt(2) * result / num_samples


@dataclass
class AveragedAcquisition:
    """Software Demodulation.

    Every readout pulse triggers an acquisition, where the 16384 i and q
    samples of the waveform acquired by the ADC are saved into a
    dedicated memory within the FPGA. This is what qblox calls *scoped
    acquisition*. The results of multiple shots are averaged in this
    memory, and cannot be retrieved independently. The resulting
    waveforms averages (i and q) are then demodulated and integrated in
    software (and finally divided by the number of samples). Since
    Software Demodulation relies on the data of the scoped acquisition
    and that data is the average of all acquisitions, **only one readout
    pulse per qubit is supported**, so that the averages all correspond
    to reading the same quantum state.
    """

    results: dict
    """Data returned by qblox acquisition."""
    duration: int
    """Duration of the readout pulse."""
    frequency: int
    """Frequency of the readout pulse used for demodulation."""

    i: Optional[List[float]] = None
    q: Optional[List[float]] = None

    @property
    def scope(self):
        return self.results["acquisition"]["scope"]

    @property
    def raw_i(self):
        """Average of the i waveforms for every readout pulse."""
        return np.array(self.scope["path0"]["data"][0 : self.duration])

    @property
    def raw_q(self):
        """Average of the q waveforms for every readout pulse."""
        return np.array(self.scope["path1"]["data"][0 : self.duration])

    @property
    def data(self):
        """Acquisition data to be returned to the platform.

        Ignores the data available in acquisition results and returns
        only i and q voltages.
        """
        # TODO: to be updated once the functionality of ExecutionResults is extended
        if self.i is None or self.q is None:
            self.i, self.q = demodulate(self.raw_i, self.raw_q, self.frequency)
        return (self.i, self.q)


@dataclass
class DemodulatedAcquisition:
    """Hardware Demodulation.

    With hardware demodulation activated, the FPGA can demodulate,
    integrate (average over time), and classify each shot individually,
    saving the results on separate bins. The raw data of each
    acquisition continues to be averaged as with software modulation, so
    there is no way to access the raw data of each shot (unless executed
    one shot at a time). The FPGA uses fixed point arithmetic for the
    demodulation and integration; if the power level of the signal at
    the input port is low (the minimum resolution of the ADC is 240uV)
    rounding precission errors can accumulate and render wrong results.
    It is advisable to have a power level at least higher than 5mV.
    """

    scope: dict
    """Data returned by scope qblox acquisition."""
    bins: dict
    """Binned acquisition data returned by qblox."""
    duration: int
    """Duration of the readout pulse."""

    @property
    def raw(self):
        return self.scope["acquisition"]["scope"]

    @property
    def integration(self):
        return self.bins["integration"]

    @property
    def shots_i(self):
        """I-component after demodulating and integrating every shot
        waveform."""
        return np.array(self.integration["path0"]) / self.duration

    @property
    def shots_q(self):
        """Q-component after demodulating and integrating every shot
        waveform."""
        return np.array(self.integration["path1"]) / self.duration

    @property
    def raw_i(self):
        """Average of the raw i waveforms for every readout pulse."""
        return np.array(self.raw["path0"]["data"][0 : self.duration])

    @property
    def raw_q(self):
        """Average of the raw q waveforms for every readout pulse."""
        return np.array(self.raw["path1"]["data"][0 : self.duration])

    @property
    def classified(self):
        """List with the results of demodulating, integrating and classifying
        every shot."""
        return np.array(self.bins["threshold"])

    @property
    def data(self):
        """Acquisition data to be returned to the platform.

        Ignores the data available in acquisition results and returns
        only i and q voltages.
        """
        # TODO: to be updated once the functionality of ExecutionResults is extended
        return (self.shots_i, self.shots_q, self.classified)
