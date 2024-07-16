import numpy as np
from qblox_instruments.qcodes_drivers.sequencer import Sequencer as QbloxSequencer

from qibolab.instruments.qblox.q1asm import Program
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.sweeper import Parameter, Sweeper

SAMPLING_RATE = 1
"""Sampling rate for qblox instruments in GSps."""


class WaveformsBuffer:
    """A class to represent a buffer that holds the unique waveforms used by a
    sequencer.

    Attributes:
        unique_waveforms (list): A list of unique Waveform objects.
        available_memory (int): The amount of memory available expressed in numbers of samples.
    """

    SIZE: int = 16383

    class NotEnoughMemory(Exception):
        """An error raised when there is not enough memory left to add more
        waveforms."""

    class NotEnoughMemoryForBaking(Exception):
        """An error raised when there is not enough memory left to bake
        pulses."""

    def __init__(self):
        """Initialises the buffer with an empty list of unique waveforms."""
        self.unique_waveforms: list = []  # Waveform
        self.available_memory: int = WaveformsBuffer.SIZE

    def add_waveforms(
        self, pulse: Pulse, hardware_mod_en: bool, sweepers: list[Sweeper]
    ):
        """Adds a pair of i and q waveforms to the list of unique waveforms.

        Waveforms are added to the list if they were not there before.
        Each of the waveforms (i and q) is processed individually.

        Args:
            waveform_i (Waveform): A Waveform object containing the samples of the real component of the pulse wave.
            waveform_q (Waveform): A Waveform object containing the samples of the imaginary component of the pulse wave.

        Raises:
            NotEnoughMemory: If the memory needed to store the waveforms in more than the memory avalible.
        """
        pulse_copy = pulse.copy()
        for sweeper in sweepers:
            if sweeper.pulses and sweeper.parameter == Parameter.amplitude:
                if pulse in sweeper.pulses:
                    pulse_copy.amplitude = 1

        baking_required = False
        for sweeper in sweepers:
            if sweeper.pulses and sweeper.parameter == Parameter.duration:
                if pulse in sweeper.pulses:
                    baking_required = True
                    values = sweeper.get_values(pulse.duration)

        if not baking_required:
            if hardware_mod_en:
                waveform_i, waveform_q = pulse_copy.envelope_waveforms(SAMPLING_RATE)
            else:
                waveform_i, waveform_q = pulse_copy.modulated_waveforms(SAMPLING_RATE)

            pulse.waveform_i = waveform_i
            pulse.waveform_q = waveform_q

            if (
                waveform_i not in self.unique_waveforms
                or waveform_q not in self.unique_waveforms
            ):
                memory_needed = 0
                if not waveform_i in self.unique_waveforms:
                    memory_needed += len(waveform_i)
                if not waveform_q in self.unique_waveforms:
                    memory_needed += len(waveform_q)

                if self.available_memory >= memory_needed:
                    if not waveform_i in self.unique_waveforms:
                        self.unique_waveforms.append(waveform_i)
                    if not waveform_q in self.unique_waveforms:
                        self.unique_waveforms.append(waveform_q)
                    self.available_memory -= memory_needed
                else:
                    raise WaveformsBuffer.NotEnoughMemory
        else:
            pulse.idx_range = self.bake_pulse_waveforms(
                pulse_copy, values, hardware_mod_en
            )

    def bake_pulse_waveforms(
        self, pulse: Pulse, values: list(), hardware_mod_en: bool
    ):  # bake_pulse_waveforms(self, pulse: Pulse, values: list(int), hardware_mod_en: bool):
        """Generates and stores a set of i and q waveforms required for a pulse
        duration sweep.

        These waveforms are generated and stored in a predefined order so that they can later be retrieved within the
        sweeper q1asm code. It bakes pulses from as short as 1ns, padding them at the end with 0s if required so that
        their length is a multiple of 4ns. It also supports the modulation of the pulse both in hardware (default)
        or software.
        With no other pulses stored in the sequencer memory, it supports values up to range(1, 126) for regular pulses and
        range(1, 180) for flux pulses.

        Args:
            pulse (:class:`qibolab.pulses.Pulse`): The pulse to be swept.
            values (list(int)): The list of values to sweep the pulse duration with.
            hardware_mod_en (bool): If set to True the pulses are assumed to be modulated in hardware and their
                envelope waveforms are uploaded; if False, software modulated waveforms are uploaded.

        Returns:
            idx_range (numpy.ndarray): An array with the indices of the set of pulses. For each pulse duration in
                `values` the i component is saved in the next avalable index, followed by the q component. For flux
                pulses, since both i and q components are equal, they are only saved once.

        Raises:
            NotEnoughMemory: If the memory needed to store the waveforms in more than the memory avalible.
        """
        # In order to generate waveforms for each duration value, the pulse will need to be modified.
        values = np.round(values).astype(int)
        # To avoid any conflicts, make a copy of the pulse first.
        pulse_copy = pulse.copy()

        # there may be other waveforms stored already, set first index as the next available
        first_idx = len(self.unique_waveforms)

        if pulse.type == PulseType.FLUX or pulse.type == PulseType.COUPLERFLUX:
            # for flux pulses, store i waveforms
            idx_range = np.arange(first_idx, first_idx + len(values), 1)

            for duration in values:
                pulse_copy.duration = duration
                if hardware_mod_en:
                    waveform = pulse_copy.envelope_waveform_i(SAMPLING_RATE)
                else:
                    waveform = pulse_copy.modulated_waveform_i(SAMPLING_RATE)

                padded_duration = int(np.ceil(duration / 4)) * 4
                memory_needed = padded_duration
                padding = np.zeros(padded_duration - duration)
                waveform.data = np.append(waveform.data, padding)

                if self.available_memory >= memory_needed:
                    self.unique_waveforms.append(waveform)
                    self.available_memory -= memory_needed
                else:
                    raise WaveformsBuffer.NotEnoughMemoryForBaking
        else:
            # for any other pulse type, store both i and q waveforms
            idx_range = np.arange(first_idx, first_idx + len(values) * 2, 2)

            for duration in values:
                pulse_copy.duration = duration
                if hardware_mod_en:
                    waveform_i, waveform_q = pulse_copy.envelope_waveforms(
                        SAMPLING_RATE
                    )
                else:
                    waveform_i, waveform_q = pulse_copy.modulated_waveforms(
                        SAMPLING_RATE
                    )

                padded_duration = int(np.ceil(duration / 4)) * 4
                memory_needed = padded_duration * 2
                padding = np.zeros(padded_duration - duration)
                waveform_i.data = np.append(waveform_i.data, padding)
                waveform_q.data = np.append(waveform_q.data, padding)

                if self.available_memory >= memory_needed:
                    self.unique_waveforms.append(waveform_i)
                    self.unique_waveforms.append(waveform_q)
                    self.available_memory -= memory_needed
                else:
                    raise WaveformsBuffer.NotEnoughMemoryForBaking

        return idx_range


class Sequencer:
    """A class to extend the functionality of qblox_instruments Sequencer.

    A sequencer is a hardware component synthesised in the instrument FPGA, responsible for fetching waveforms from
    memory, pre-processing them, sending them to the DACs, and processing the acquisitions from the ADCs (QRM modules).
    https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/documentation/sequencer.html

    This class extends the sequencer functionality by holding additional data required when
    processing a pulse sequence:

    - the sequencer number,
    - the sequence of pulses to be played,
    - a buffer of unique waveforms, and
    - the four components of the sequence file:

        - waveforms dictionary
        - acquisition dictionary
        - weights dictionary
        - program

    Attributes:
        device (QbloxSequencer): A reference to the underlying `qblox_instruments.qcodes_drivers.sequencer.Sequencer`
            object. It can be used to access other features not directly exposed by this wrapper.
            https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/documentation/sequencer.html
        number (int): An integer between 0 and 5 that identifies the number of the sequencer.
        pulses (PulseSequence): The sequence of pulses to be played by the sequencer.
        waveforms_buffer (WaveformsBuffer): A buffer of unique waveforms to be played by the sequencer.
        waveforms (dict): A dictionary containing the waveforms to be played by the sequencer in qblox format.
        acquisitions (dict): A dictionary containing the list of acquisitions to be made by the sequencer in qblox
            format.
        weights (dict): A dictionary containing the list of weights to be used by the sequencer when demodulating
            and integrating the response, in qblox format.
        program (str): The pseudo assembly (q1asm) program to be executed by the sequencer.
            https://qblox-qblox-instruments.readthedocs-hosted.com/en/master/documentation/sequencer.html#instructions

        qubit (str): The id of the qubit associated with the sequencer, if there is only one.
    """

    def __init__(self, number: int):
        """Initialises the sequencer.

        All class attributes are defined and initialised.
        """

        self.device: QbloxSequencer = None
        self.number: int = number
        self.pulses: PulseSequence = PulseSequence()
        self.waveforms_buffer: WaveformsBuffer = WaveformsBuffer()
        self.waveforms: dict = {}
        self.acquisitions: dict = {}
        self.weights: dict = {}
        self.program: Program = Program()
        self.qubit = None  # self.qubit: int | str = None
        self.coupler = None  # self.coupler: int | str = None
