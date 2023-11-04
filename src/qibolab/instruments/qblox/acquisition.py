from dataclasses import dataclass, field

import numpy as np


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

    raw: field(default_factory=dict)
    """Tuples with the averages of the i and q waveforms for every readout pulse ([i samples], [q samples])
    demodulated_integrated: field(default_factory=dict)

    The data for a specific reaout pulse can be obtained either with:
    - `raw[ro_pulse.serial]`
    - `raw[ro_pulse.qubit]`
    """

    demodulated_integrated: dict = field(default_factory=dict)
    """Tuples with the results of demodulating and integrating (averaging over time) the average of the
    waveforms for every pulse: ``(amplitude[V], phase[rad], i[V], q[V])``

    The data for a specific readout pulse can be obtained either with:
    - `averaged_demodulated_integrated[ro_pulse.serial]`
    - `averaged_demodulated_integrated[ro_pulse.qubit]`
    Or directly with (see __getitem__):
    - `self[ro_pulse.serial]`
    - `self[ro_pulse.qubit]`
    """

    @property
    def data(self):
        """Acquisition data to be returned to the platform.

        Ignores the data available in acquisition results and returns only i and q voltages.
        """
        # TODO: to be updated once the functionality of ExecutionResults is extended
        # (i, q)
        return {serial: value[2:] for serial, value in self.demodulated_integrated.items()}

    def __getitem__(self, key):
        """Default Results: Averaged Demodulated Integrated"""
        return self.demodulated_integrated[key]

    def register(self, scope, pulse, duration):
        self.raw[pulse.qubit] = self.raw[pulse.serial] = (
            raw["path0"]["data"][0:duration],
            raw["path1"]["data"][0:duration],
        )

    def demodulate(self, scope, frequency, duration):
        """Demodulates and integrates the acquired pulse."""
        # input_vec_i = np.array(acquisition_results["acquisition"]["scope"]["path0"]["data"][0: duration])

        # DOWN Conversion
        input_vec_i = np.array(scope["path0"]["data"][0:duration])
        input_vec_q = np.array(scope["path1"]["data"][0:duration])
        # qblox does not remove the offsets in hardware
        input_vec_i -= np.mean(input_vec_i)
        input_vec_q -= np.mean(input_vec_q)

        modulated_i = input_vec_i
        modulated_q = input_vec_q

        num_samples = modulated_i.shape[0]
        time = np.arange(num_samples) / PulseShape.SAMPLING_RATE
        cosalpha = np.cos(2 * np.pi * frequency * time)
        sinalpha = np.sin(2 * np.pi * frequency * time)
        demod_matrix = np.sqrt(2) * np.array([[cosalpha, sinalpha], [-sinalpha, cosalpha]])
        result = []
        for it, t, ii, qq in zip(np.arange(modulated_i.shape[0]), time, modulated_i, modulated_q):
            result.append(demod_matrix[:, :, it] @ np.array([ii, qq]))
        i, q = np.mean(np.array(result), axis=0)

        data = (np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q)
        self.demodulated_integrated[pulse.qubit] = self.demodulated_integrated[pulse.serial] = data


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

    integrated_averaged: dict = field(default_factory=dict)
    """a dictionary containing tuples with the results of demodulating and integrating (averaging over time)
    each shot waveform and then averaging of the many shots: ``(amplitude[V], phase[rad], i[V], q[V])``
    """
    integrated_binned: dict = field(default_factory=dict)
    """a dictionary containing tuples of lists with the results of demodulating and integrating
    every shot waveform: ``([amplitudes[V]], [phases[rad]], [is[V]], [qs[V]])``
    """
    integrated_classified_binned: dict = field(default_factory=dict)
    """a dictionary containing lists with the results of demodulating, integrating and
    classifying every shot: ``([states[0 or 1]])``
    """

    averaged: AveragedAcquisition = field(default_factory=lambda: AveragedAcquisition())
    """If the number of readout pulses per qubit is only one, then the
    :class:`qibolab.instruments.qblox.AveragedAcquisition` data are also provided.
    """

    @property
    def data(self):
        """Acquisition data to be returned to the platform.

        Ignores the data available in acquisition results and returns only i and q voltages.
        """
        # TODO: to be updated once the functionality of ExecutionResults is extended
        _data = {}
        for serial, value in self.integrated_binned.items():
            _data[serial] = (value[2], value[3], np.array(self.integrated_classified_binned[serial]))
        return _data

    def __getitem__(self, key):
        return self.integrated_averaged[key]

    @property
    def probability(self):
        """Dictionary containing the frequency of state 1 measurements.

        Calculated as number of shots classified as 1 / total number of shots.
        """
        return {serial: np.mean(value) for serial, value in self.integrated_binned.items()}

    def register(self, bins, pulse, duration):
        """Calculates average by dividing the integrated results by the number of samples acquired."""
        i = np.mean(np.array(bins["integration"]["path0"])) / duration
        q = np.mean(np.array(bins["integration"]["path1"])) / duration
        data = (np.sqrt(i**2 + q**2), np.arctan2(q, i), i, q)
        self.integrated_averaged[pulse.qubit] = self.integrated_averaged[pulse.serial] = data

        # Save individual shots
        integration = bins["integration"]
        shots_i = np.array(integration["path0"]) / duration
        shots_q = np.array(integration["path1"]) / duration
        data = (np.sqrt(shots_i**2 + shots_q**2), np.arctan2(shots_q, shots_i), shots_i, shots_q)
        self.integrated_binned[pulse.qubit] = self.integrated_binned[pulse.serial] = data

        self.integrated_classified_binned[pulse.qubit] = self.integrated_classified_binned[pulse.serial] = bins[
            "threshold"
        ]
