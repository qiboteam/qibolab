import operator
from dataclasses import dataclass
from typing import Union

import numpy as np
from icarusq_rfsoc_driver import IcarusQRFSoC  # pylint: disable=E0401
from icarusq_rfsoc_driver.rfsoc_settings import TRIGGER_MODE  # pylint: disable=E0401
from qibo.config import log

from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.instruments.abstract import Controller
from qibolab.pulses import Pulse, PulseSequence
from qibolab.qubits import Qubit, QubitId
from qibolab.result import average, average_iq, collect
from qibolab.sweeper import Parameter, Sweeper, SweeperType

DAC_SAMPLNG_RATE_MHZ = 5898.24
ADC_SAMPLNG_RATE_MHZ = 1966.08
ICARUSQ_PORT = 8080


@dataclass
class RFSOCPort:
    name: str
    dac: int = None
    adc: int = None


class RFSOC(Controller):
    """Driver for the IcarusQ RFSoC socket-based implementation."""

    PortType = RFSOCPort

    def __init__(
        self,
        name,
        address,
        delay_samples_offset_dac: int = 0,
        delay_samples_offset_adc: int = 0,
    ):
        super().__init__(name, address)

        self.channel_delay_offset_dac = delay_samples_offset_dac
        self.channel_delay_offset_adc = delay_samples_offset_adc

    def connect(self):
        self.device = IcarusQRFSoC(self.address, ICARUSQ_PORT)

        for dac in range(self.device.dac_nchannels):
            self.device.dac[dac].delay = self.channel_delay_offset_dac
        for adc in range(self.device.adc_nchannels):
            self.device.adc[adc].delay = self.channel_delay_offset_adc

        self.device.set_adc_trigger_mode(TRIGGER_MODE.SLAVE)
        ver = self.device.get_server_version()
        log.info(f"Connected to {self.name}, version: {ver}")

    def setup(self):
        pass

    @property
    def sampling_rate(self):
        return self.device.dac_sampling_rate / 1e3  # MHz to GHz

    def play(
        self,
        qubits: dict[QubitId, Qubit],
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
        dac_sr_ghz = dac_sampling_rate / 1e9

        # We iterate over the seuence of pulses and generate the waveforms for each type of pulses
        for ch, seq in sequence.items():
            for pulse in seq:
                # pylint: disable=no-member
                # FIXME: ignore complaint about non-existent ports and _ports properties, until we upgrade this driver to qibolab 0.2
                if pulse.channel not in self._ports:
                    continue

                dac = self.ports(pulse.channel).dac
                start = int(pulse.start * 1e-9 * dac_sampling_rate)
                i_env = pulse.envelope_waveform_i(dac_sr_ghz).data
                q_env = pulse.envelope_waveform_q(dac_sr_ghz).data

                # Flux pulses
                # TODO: Add envelope support for flux pulses
                if "flux" in ch:
                    wfm = i_env
                    end = start + len(wfm)

                # Qubit drive microwave signals
                elif "drive" in ch:
                    end = start + len(i_env)
                    t = np.arange(start, end) / dac_sampling_rate
                    cosalpha = np.cos(
                        2 * np.pi * pulse.frequency * t + pulse.relative_phase
                    )
                    sinalpha = np.sin(
                        2 * np.pi * pulse.frequency * t + pulse.relative_phase
                    )
                    wfm = i_env * sinalpha + q_env * cosalpha

                elif "probe" in ch:
                    # For readout pulses, we move the corresponding DAC/ADC pair to the start of the pulse to save memory
                    # This locks the phase of the readout in the demodulation
                    adc = self.ports(pulse.channel).adc
                    start = 0

                    end = start + len(i_env)
                    t = np.arange(start, end) / dac_sampling_rate
                    cosalpha = np.cos(
                        2 * np.pi * pulse.frequency * t + pulse.relative_phase
                    )
                    sinalpha = np.sin(
                        2 * np.pi * pulse.frequency * t + pulse.relative_phase
                    )
                    wfm = i_env * sinalpha + q_env * cosalpha

                    # First we convert the pulse starting time to number of ADC samples
                    # Then, we convert this number to the number of ADC clock cycles (8 samples per clock cycle)
                    # Next, we raise it to the next nearest integer to prevent an overlap between drive and readout pulses
                    # Finally, we ensure that the number is even for the DAC delay conversion
                    delay_start_adc = int(
                        int(
                            np.ceil(
                                self.device.adc_sampling_rate
                                * 1e6
                                * pulse.start
                                * 1e-9
                                / 8
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

                end = start + len(wfm)
                waveform_array[dac][start:end] += self.device.dac_max_amplitude * wfm
                dac_end_addr[dac] = max(end >> 4, dac_end_addr[dac])

        payload = [
            (dac, wfm, dac_end_addr[dac])
            for dac, wfm in waveform_array.items()
            if dac_end_addr[dac] != 0
        ]
        self.device.upload_waveform(payload)

    def disconnect(self):
        if self.is_connected:
            self.device.sock.close()

    def sweep(self):
        pass


class RFSOC_RO(RFSOC):
    """IcarusQ RFSoC attached with readout capability."""

    available_sweep_parameters = {
        Parameter.amplitude,
        Parameter.duration,
        Parameter.frequency,
        Parameter.relative_phase,
    }

    def __init__(
        self,
        name,
        address,
        delay_samples_offset_dac: int = 0,
        delay_samples_offset_adc: int = 0,
        adcs_to_read: list[int] = [],
    ):
        super().__init__(
            name, address, delay_samples_offset_dac, delay_samples_offset_adc
        )
        self.adcs_to_read = adcs_to_read

    def connect(self):
        super().connect()
        self.device.init_qunit()
        self.device.set_adc_trigger_mode(TRIGGER_MODE.MASTER)

    def play(
        self,
        qubits: dict[QubitId, Qubit],
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
        readout_pulses = sequence.probe_pulses
        readout_qubits = [pulse.qubit for pulse in readout_pulses]

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

            qunit_mapping = {
                ro_pulse.qubit: ro_pulse.serial for ro_pulse in readout_pulses
            }

            if options.averaging_mode is not AveragingMode.SINGLESHOT:
                res = {
                    qunit_mapping[qunit]: average_iq(i, q)
                    for qunit, (i, q) in raw.items()
                }
            else:
                res = {
                    qunit_mapping[qunit]: average_iq(i, q)
                    for qunit, (i, q) in raw.items()
                }
            # Temp fix for readout pulse sweepers, to be removed with IcarusQ v2
            for ro_pulse in readout_pulses:
                res[ro_pulse.qubit] = res[ro_pulse.serial]
            return res

        elif options.acquisition_type is AcquisitionType.DISCRIMINATION:
            self.device.set_adc_trigger_mode(1)
            self.device.set_qunit_mode(1)
            res = self.device.start_qunit_acquisition(options.nshots, readout_qubits)
            # Temp fix for readout pulse sweepers, to be removed with IcarusQ v2
            for ro_pulse in readout_pulses:
                res[ro_pulse.qubit] = res[ro_pulse.serial]
            return res

    def process_readout_signal(
        self,
        adc_raw_data: dict[int, np.ndarray],
        sequence: list[Pulse],
        qubits: dict[QubitId, Qubit],
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

            i = np.dot(raw_signal, cos)
            q = np.dot(raw_signal, sin)
            singleshot = collect(i, q)
            results[readout_pulse.serial] = (
                average(singleshot)
                if options.averaging_mode is not AveragingMode.SINGLESHOT
                else singleshot
            )
            # Temp fix for readout pulse sweepers, to be removed with IcarusQ v2
            results[readout_pulse.qubit] = results[readout_pulse.serial]

        return results

    def sweep(
        self,
        qubits: dict[QubitId, Qubit],
        couplers,
        sequence: PulseSequence,
        options: ExecutionParameters,
        *sweeper: Sweeper,
    ):
        # Record pulse values before sweeper modification
        bsv = []
        for sweep in sweeper:
            if sweep.parameter not in self.available_sweep_parameters:
                raise NotImplementedError(
                    "Sweep parameter requested not available", param_name
                )

            param_name = sweep.parameter.name.lower()
            base_sweeper_values = [getattr(pulse, param_name) for pulse in sweep.pulses]
            bsv.append(base_sweeper_values)

        res = self._sweep_recursion(qubits, couplers, sequence, options, *sweeper)

        # Reset pulse values back to original values
        for sweep, base_sweeper_values in zip(sweeper, bsv):
            param_name = sweep.parameter.name.lower()
            for pulse, value in zip(sweep.pulses, base_sweeper_values):
                setattr(pulse, param_name, value)

                # Since the sweeper will modify the readout pulse serial, we collate the results with the qubit number.
                # This is only for qibocal compatiability and will be removed with IcarusQ v2.
                # FIXME: if this was required, now it's completely broken, since it
                # isn't possible to identify the pulse channel from the pulse itself
                # (nor it should be needed)
                # if pulse.type is PulseType.READOUT:
                #     res[pulse.serial] = res[pulse.qubit]

        return res

    def _sweep_recursion(
        self,
        qubits: dict[QubitId, Qubit],
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
        sweeper_op = _sweeper_operation.get(sweep.type)
        ret = {}

        for value in sweep.values:
            for idx, pulse in enumerate(sweep.pulses):
                base = base_sweeper_values[idx]
                setattr(pulse, param_name, sweeper_op(value, base))

            self.merge_sweep_results(
                ret,
                self._sweep_recursion(
                    qubits, couplers, sequence, options, *sweeper[1:]
                ),
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


_sweeper_operation = {
    SweeperType.ABSOLUTE: lambda value, base: value,
    SweeperType.OFFSET: operator.add,
    SweeperType.FACTOR: operator.mul,
}
