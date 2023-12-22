from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
from icarusq_rfsoc_driver import IcarusQRFSoC  # pylint: disable=E0401
from icarusq_rfsoc_driver.rfsoc_settings import TRIGGER_MODE  # pylint: disable=E0401
from qibo.config import log
from scipy.signal.windows import gaussian, hamming

from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.instruments.abstract import Controller
from qibolab.instruments.port import Port
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.qubits import Qubit, QubitId
from qibolab.result import IntegratedResults, SampleResults
from qibolab.sweeper import Parameter, Sweeper, SweeperType


@dataclass
class RFSOCPort(Port):
    name: str
    dac: int = None
    adc: int = None


class RFSOC(Controller):
    """Driver for the IcarusQ RFSoC socket-based implementation."""

    PortType = RFSOCPort

    def __init__(self, name, address, port=8080):
        super().__init__(name, address)
        self.device = IcarusQRFSoC(address, port)

        self.channel_delay_offset_dac = 0
        self.channel_delay_offset_adc = 0

    def setup(
        self,
        dac_sampling_rate: float,
        adc_sampling_rate: float,
        delay_samples_offset_dac: int = 0,
        delay_samples_offset_adc: int = 0,
        analog_settings: List["dict[str, int]"] = [],
        **kwargs,
    ):
        """Performs the setup for the IcarusQ RFSoC.

        Arguments:
            dac_sampling_rate (float): Sampling rate of the 16 DACs in MHz.
            adc_sampling_rate (float): Sampling rate of the 16 ADCs in MHz.
            delay_samples_offset_dac (int): Number of clock cycles (per 16 DAC samples) to delay all DAC playback
            delay_samples_offset_adc (int): Number of clock cycles (per 8 ADC samples) to delay all ADC acquistion
            analog_settings (list): List of analog settings per DAC/ADC channel to set
        """

        self.device.dac_sampling_rate = dac_sampling_rate
        self.device.adc_sampling_rate = adc_sampling_rate

        self.channel_delay_offset_dac = delay_samples_offset_dac
        self.channel_delay_offset_adc = delay_samples_offset_adc

        for dac in range(self.device.dac_nchannels):
            self.device.dac[dac].delay = delay_samples_offset_dac
        for adc in range(self.device.adc_nchannels):
            self.device.adc[adc].delay = delay_samples_offset_adc
        self.device.set_adc_trigger_mode(TRIGGER_MODE.SLAVE)

        for channel_settings in analog_settings:
            self.device.set_channel_analog_settings(**channel_settings)

    def play(
        self,
        qubits: Dict[QubitId, Qubit],
        couplers,
        sequence: PulseSequence,
        options: ExecutionParameters,
    ):
        """Plays the given pulse sequence without acquisition.

        Arguments:
            qubits (dict): Dictionary of qubit IDs mapped to qubit objects.
            sequence (PulseSequence): Pulse sequence to be played on this instrument.
            options (ExecutionParameters): Execution parameters for readout and repetition.
        """

        waveform_array = {dac.id: np.zeros(dac.max_samples) for dac in self.device.dac}

        dac_end_addr = {dac.id: 0 for dac in self.device.dac}
        dac_sampling_rate = self.device.dac_sampling_rate * 1e6

        # We iterate over the seuence of pulses and generate the waveforms for each type of pulses
        for pulse in sequence.pulses:
            if pulse.channel not in self._ports:
                continue

            dac = self.ports(pulse.channel).dac
            start = int(pulse.start * 1e-9 * dac_sampling_rate)
            end = int((pulse.start + pulse.duration) * 1e-9 * dac_sampling_rate)
            num_samples = end - start

            # Flux pulses
            # TODO: Add envelope support for flux pulses
            if pulse.type == PulseType.FLUX:
                wfm = np.ones(num_samples)

            # Qubit drive microwave signals
            elif pulse.type == PulseType.DRIVE:
                t = np.arange(start, end) / dac_sampling_rate
                wfm = np.sin(2 * np.pi * pulse.frequency * t + pulse.relative_phase)

                # Currently we only support DRAG pulses
                if pulse.shape.name == "DRAG":
                    sigma = num_samples / pulse.shape.rel_sigma
                    beta = pulse.shape.beta
                    real = gaussian(num_samples, sigma)
                    img = (
                        -beta
                        * (np.arange(num_samples) - num_samples / 2)
                        * real
                        / sigma**2
                    )

                    wfm *= real
                    wfm += img * np.cos(
                        2 * np.pi * pulse.frequency * t + pulse.relative_phase
                    )

                else:
                    wfm *= hamming(wfm.shape[0])

            elif pulse.type == PulseType.READOUT:
                # For readout pulses, we move the corresponding DAC/ADC pair to the start of the pulse to save memory
                # This locks the phase of the readout in the demodulation
                adc = self.ports(pulse.channel).adc
                start = 0
                end = int(pulse.duration * 1e-9 * dac_sampling_rate)

                t = np.arange(start, end) / dac_sampling_rate
                wfm = np.sin(2 * np.pi * pulse.frequency * t + pulse.relative_phase)

                # First we convert the pulse starting time to number of ADC samples
                # Then, we convert this number to the number of ADC clock cycles (8 samples per clock cycle)
                # Next, we raise it to the next nearest integer to prevent an overlap between drive and readout pulses
                # Finally, we ensure that the number is even for the DAC delay conversion
                delay_start_adc = int(
                    int(
                        np.ceil(
                            self.device.adc_sampling_rate * 1e6 * pulse.start * 1e-9 / 8
                        )
                        / 2
                    )
                    * 2
                )

                # For the DAC, currently the sampling rate is 3x higher than the ADC
                # The number of clock cycles is 16 samples per clock cycle
                # Hence, we multiply the adc delay clock cycles by 1.5x to align the DAC/ADC pair
                delay_start_dac = int(delay_start_adc * 1.5)

                self.device.dac[dac].delay = (
                    delay_start_dac + self.channel_delay_offset_dac
                )
                self.device.adc[adc].delay = (
                    delay_start_adc + self.channel_delay_offset_adc
                )
                # ADC0 complete marks the end of acquisition, so we also need to move ADC0
                self.device.adc[0].delay = (
                    delay_start_adc + self.channel_delay_offset_adc
                )

                if (
                    options.acquisition_type is AcquisitionType.DISCRIMINATION
                    or AcquisitionType.INTEGRATION
                ):
                    self.device.program_qunit(
                        readout_frequency=pulse.frequency,
                        readout_time=pulse.duration * 1e-9,
                        qunit=pulse.qubit,
                    )

            waveform_array[dac][start:end] += (
                self.device.dac_max_amplitude * pulse.amplitude * wfm
            )
            dac_end_addr[dac] = max(end >> 4, dac_end_addr[dac])

        payload = [
            (dac, wfm, dac_end_addr[dac])
            for dac, wfm in waveform_array.items()
            if dac_end_addr[dac] != 0
        ]
        self.device.upload_waveform(payload)

    def play_sequences(
        self,
        qubits: Dict[QubitId, Qubit],
        couplers,
        sequences: List[PulseSequence],
        options: ExecutionParameters,
    ):
        pass

    def connect(self):
        """Currently we only connect to the board when we have to send a
        command."""
        # Request the version from the board
        ver = self.device.get_server_version()
        log.info(f"Connected to {self.name}, version: {ver}")

    def start(self):
        pass

    def stop(self):
        pass

    def disconnect(self):
        pass

    def sweep(self):
        pass


class RFSOC_RO(RFSOC):
    """IcarusQ RFSoC attached with readout capability."""

    available_sweep_parameters = {
        Parameter.amplitude,
        Parameter.duration,
        Parameter.frequency,
        Parameter.relative_phase,
        Parameter.start,
    }

    def __init__(self, name, address):
        super().__init__(name, address)

        self.qubit_adc_map = {}
        self.adcs_to_read: List[int] = None

    def setup(
        self,
        dac_sampling_rate: float,
        adc_sampling_rate: float,
        delay_samples_offset_dac: int = 0,
        delay_samples_offset_adc: int = 0,
        analog_settings: List["dict[str, int]"] = [],
        adcs_to_read: List[int] = [],
        **kwargs,
    ):
        """Setup the board and assign ADCs to be read.

        Arguments:
            dac_sampling_rate (float): Sampling rate of the 16 DACs in MHz.
            adc_sampling_rate (float): Sampling rate of the 16 ADCs in MHz.
            delay_samples_offset_dac (int): Number of clock cycles (per 16 DAC samples) to delay all DAC playback.
            delay_samples_offset_adc (int): Number of clock cycles (per 8 ADC samples) to delay all ADC acquistion.
            analog_settings (list): List of analog settings per DAC/ADC channel to set.
            adcs_to_read (list[int]): List of ADC channels to be read.
        """

        super().setup(
            dac_sampling_rate,
            adc_sampling_rate,
            delay_samples_offset_dac,
            delay_samples_offset_adc,
            analog_settings,
        )
        self.adcs_to_read = adcs_to_read

        self.device.init_qunit()
        self.device.set_adc_trigger_mode(TRIGGER_MODE.MASTER)

    def play(
        self,
        qubits: Dict[QubitId, Qubit],
        couplers,
        sequence: PulseSequence,
        options: ExecutionParameters,
    ):
        """Plays the pulse sequence on the IcarusQ RFSoC and awaits acquisition
        at the end.

        Arguments:
            qubits (dict): Dictionary of qubit IDs mapped to qubit objects.
            sequence (PulseSequence): Pulse sequence to be played on this instrument.
            options (ExecutionParameters): Object representing acquisition type and number of shots.
        """
        super().play(qubits, couplers, sequence, options)
        self.device.set_adc_trigger_repetition_rate(int(options.relaxation_time / 1e3))
        readout_pulses = list(
            filter(lambda pulse: pulse.type is PulseType.READOUT, sequence.pulses)
        )
        readout_qubits = {pulse.qubit for pulse in readout_pulses}

        if options.acquisition_type is AcquisitionType.RAW:
            self.device.set_adc_trigger_mode(0)
            self.device.arm_adc(self.adcs_to_read, options.nshots)
            raw = self.device.result()
            return self.process_readout_signal(raw, readout_pulses, qubits, options)

        # Currently qunit only supports single qubit readout demodulation
        elif options.acquisition_type is AcquisitionType.INTEGRATION:
            self.device.set_adc_trigger_mode(1)
            self.device.set_qunit_mode(0)
            raw = self.device.start_qunit_acquisition(options.nshots, readout_qubits)

            if options.averaging_mode is not AveragingMode.SINGLESHOT:
                res = {
                    qubit: IntegratedResults(I + 1j * Q).average
                    for qubit, (I, Q) in raw.items()
                }
            else:
                res = {
                    qubit: IntegratedResults(I + 1j * Q)
                    for qubit, (I, Q) in raw.items()
                }

            for ro_pulse in readout_pulses:
                res[ro_pulse.serial] = res[ro_pulse.qubit]
            return res

        elif options.acquisition_type is AcquisitionType.DISCRIMINATION:
            self.device.set_adc_trigger_mode(1)
            self.device.set_qunit_mode(1)
            raw = self.device.start_qunit_acquisition(options.nshots)
            res = {qubit: SampleResults(states) for qubit, states in raw.items()}
            for ro_pulse in readout_pulses:
                res[ro_pulse.serial] = res[ro_pulse.qubit]
            return res

    def play_sequences(
        self,
        qubits: Dict[QubitId, Qubit],
        couplers,
        sequences: List[PulseSequence],
        options: ExecutionParameters,
    ):
        """Wrapper for play."""
        return [
            self.play(qubits, couplers, sequence, options) for sequence in sequences
        ]

    def process_readout_signal(
        self,
        adc_raw_data: Dict[int, np.ndarray],
        sequence: List[Pulse],
        qubits: Dict[QubitId, Qubit],
        options: ExecutionParameters,
    ):
        """Processes the raw signal from the ADC into IQ values."""

        adc_sampling_rate = self.device.adc_sampling_rate * 1e6
        t = np.arange(self.device.adc_sample_size) / adc_sampling_rate
        results = {}

        for readout_pulse in sequence:
            qubit = qubits[readout_pulse.qubit]
            _, adc = qubit.readout.ports

            raw_signal = adc_raw_data[adc]
            sin = np.sin(2 * np.pi * readout_pulse.frequency * t)
            cos = np.sin(2 * np.pi * readout_pulse.frequency * t)

            I = np.dot(raw_signal, cos)
            Q = np.dot(raw_signal, sin)
            results[readout_pulse.qubit] = IntegratedResults(I + 1j * Q)

            if options.averaging_mode is not AveragingMode.SINGLESHOT:
                results[readout_pulse.qubit] = results[readout_pulse.serial] = results[
                    readout_pulse.qubit
                ].average
            else:
                results[readout_pulse.serial] = results[readout_pulse.qubit]

        return results

    def sweep(
        self,
        qubits: Dict[QubitId, Qubit],
        couplers,
        sequence: PulseSequence,
        options: ExecutionParameters,
        *sweeper: Sweeper,
    ):
        """Recursive python-based sweeper functionaltiy for the IcarusQ
        RFSoC."""
        if len(sweeper) == 0:
            return self.play(qubits, couplers, sequence, options)

        sweep = sweeper[0]
        param = sweep.parameter
        param_name = param.name.lower()

        if param not in self.available_sweep_parameters:
            raise NotImplementedError(
                "Sweep parameter requested not available", param_name
            )

        base_sweeper_values = [getattr(pulse, param_name) for pulse in sweep.pulses]
        sweeper_value_setter_fn = _sweeper_value_setter.get(sweep.type)
        ret = {}

        for value in sweep.values:
            for idx, pulse in enumerate(sweep.pulses):
                base = base_sweeper_values[idx]
                sweeper_value_setter_fn(pulse, param_name, value, base)

            self.merge_sweep_results(
                ret, self.sweep(qubits, couplers, sequence, options, *sweeper[1:])
            )

        return ret

    @staticmethod
    def merge_sweep_results(
        dict_a: """dict[str, Union[IntegratedResults, SampleResults]]""",
        dict_b: """dict[str, Union[IntegratedResults, SampleResults]]""",
    ) -> """dict[str, Union[IntegratedResults, SampleResults]]""":
        """Merge two dictionary mapping pulse serial to Results object.

        If dict_b has a key (serial) that dict_a does not have, simply add it,
        otherwise sum the two results

        Args:
            dict_a (dict): dict mapping ro pulses serial to qibolab res objects
            dict_b (dict): dict mapping ro pulses serial to qibolab res objects
        Returns:
            A dict mapping the readout pulses serial to qibolab results objects
        """
        for serial in dict_b:
            if serial in dict_a:
                dict_a[serial] = dict_a[serial] + dict_b[serial]
            else:
                dict_a[serial] = dict_b[serial]
        return dict_a


_sweeper_type_absolute = lambda pulse, param, value, base: setattr(pulse, param, value)
_sweeper_type_offset = lambda pulse, param, value, base: setattr(
    pulse, param, base + value
)
_sweeper_type_factor = lambda pulse, param, value, base: setattr(
    pulse, param, base * value
)
_sweeper_value_setter = {
    SweeperType.ABSOLUTE: _sweeper_type_absolute,
    SweeperType.OFFSET: _sweeper_type_offset,
    SweeperType.FACTOR: _sweeper_type_factor,
}
