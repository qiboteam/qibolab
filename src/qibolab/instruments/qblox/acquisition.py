from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def demodulate(input_i, input_q, frequency):
    """Demodulates and integrates the acquired pulse."""
    # DOWN Conversion
    # qblox does not remove the offsets in hardware
    modulated_i = input_i - np.mean(input_i)
    modulated_q = input_q - np.mean(input_q)

    num_samples = modulated_i.shape[0]
    time = np.arange(num_samples) / PulseShape.SAMPLING_RATE
    cosalpha = np.cos(2 * np.pi * frequency * time)
    sinalpha = np.sin(2 * np.pi * frequency * time)
    demod_matrix = np.sqrt(2) * np.array([[cosalpha, sinalpha], [-sinalpha, cosalpha]])
    result = []
    for it, t, ii, qq in zip(np.arange(modulated_i.shape[0]), time, modulated_i, modulated_q):
        result.append(demod_matrix[:, :, it] @ np.array([ii, qq]))
    return np.mean(np.array(result), axis=0)


@dataclass
class AveragedAcquisition:
    """Software Demodulation.

    Every readout pulse triggers an acquisition, where the 16384 i and q samples of the waveform
    acquired by the ADC are saved into a dedicated memory within the FPGA. This is what qblox calls
    *scoped acquisition*. The results of multiple shots are averaged in this memory, and cannot be
    retrieved independently. The resulting waveforms averages (i and q) are then demodulated and
    integrated in software (and finally divided by the number of samples).
    Since Software Demodulation relies on the data of the scoped acquisition and that data is the
    average of all acquisitions, **only one readout pulse per qubit is supported**, so that
    the averages all correspond to reading the same quantum state.
    """

    raw: Tuple[List[float], List[float]]
    """Tuples with the averages of the i and q waveforms for every readout pulse ([i samples], [q samples])."""

    demodulated_integrated: Tuple[float, float, float, float]
    """Tuples with the results of demodulating and integrating (averaging over time) the average of the
    waveforms for every pulse: ``(amplitude[V], phase[rad], i[V], q[V])``."""

    @property
    def data(self):
        """Acquisition data to be returned to the platform.

        Ignores the data available in acquisition results and returns only i and q voltages.
        """
        # TODO: to be updated once the functionality of ExecutionResults is extended
        return self.demodulated_integrated[2:]  # (i, q)

    @property
    def default(self):
        """Default Results: Averaged Demodulated Integrated"""
        return self.demodulated_integrated

    @classmethod
    def create(cls, data, duration, frequency):
        scope = data["acquisition"]["scope"]
        raw = (
            scope["path0"]["data"][0:duration],
            scope["path1"]["data"][0:duration],
        )
        i, q = demodulate(*raw, frequency)
        demod = (np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q)
        return cls(raw, demod)


@dataclass
class DemodulatedAcquisition:
    """Hardware Demodulation.

    With hardware demodulation activated, the FPGA can demodulate, integrate (average over time), and classify
    each shot individually, saving the results on separate bins. The raw data of each acquisition continues to
    be averaged as with software modulation, so there is no way to access the raw data of each shot (unless
    executed one shot at a time). The FPGA uses fixed point arithmetic for the demodulation and integration;
    if the power level of the signal at the input port is low (the minimum resolution of the ADC is 240uV)
    rounding precission errors can accumulate and render wrong results. It is advisable to have a power level
    at least higher than 5mV.

    The data within each of the above dictionaries, for a specific readout pulse or for the last readout
    pulse of a qubit can be retrieved either with:
    - `self.data_type_name[ro_pulse.serial]`
    - `self.data_type_name[ro_pulse.qubit]`

    And "averaged_demodulated_integrated" directly with:
    - `self[ro_pulse.serial]`
    - `self[ro_pulse.qubit]`
    """

    integrated_averaged: Tuple[float, float, float, float]
    """Tuple with the results of demodulating and integrating (averaging over time)
    each shot waveform and then averaging of the many shots: ``(amplitude[V], phase[rad], i[V], q[V])``
    """
    integrated_binned: Tuple[List[float], List[float], List[float], List[float]]
    """Tuple of lists with the results of demodulating and integrating
    every shot waveform: ``([amplitudes[V]], [phases[rad]], [is[V]], [qs[V]])``
    """
    integrated_classified_binned: List[int]
    """Lists with the results of demodulating, integrating and
    classifying every shot: ``([states[0 or 1]])``
    """

    averaged: Optional[AveragedAcquisition] = None
    """If the number of readout pulses per qubit is only one, then the
    :class:`qibolab.instruments.qblox.AveragedAcquisition` data are also provided.
    """

    @property
    def data(self):
        """Acquisition data to be returned to the platform.

        Ignores the data available in acquisition results and returns only i and q voltages.
        """
        # TODO: to be updated once the functionality of ExecutionResults is extended
        return (
            self.integrated_binned[2],
            self.integrated_binned[3],
            np.array(self.integrated_classified_binned[serial]),
        )

    @property
    def default(self):
        return self.integrated_averaged

    @property
    def probability(self):
        """Dictionary containing the frequency of state 1 measurements.

        Calculated as number of shots classified as 1 / total number of shots.
        """
        return np.mean(self.integrated_binned)

    @classmethod
    def create(self, data, pulse, duration):
        """Calculates average by dividing the integrated results by the number of samples acquired."""
        bins = data[pulse.serial]["acquisition"]["bins"]
        i = np.mean(np.array(bins["integration"]["path0"])) / duration
        q = np.mean(np.array(bins["integration"]["path1"])) / duration
        averaged = (np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q)

        # Save individual shots
        integration = bins["integration"]
        shots_i = np.array(integration["path0"]) / duration
        shots_q = np.array(integration["path1"]) / duration
        integrated = (np.sqrt(shots_i**2 + shots_q**2), np.arctan2(shots_q, shots_i), shots_i, shots_q)

        classified = bins["threshold"]

        return cls(averaged, binned, classified)
