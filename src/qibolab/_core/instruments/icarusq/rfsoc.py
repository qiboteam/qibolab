import numpy as np
from icarusq_rfsoc_driver import IcarusQRFSoC

from qibolab._core.components import AcquisitionChannel, Config, IqChannel
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.identifier import Result
from qibolab._core.instruments.awg import AWG
from qibolab._core.pulses.pulse import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers
from qibolab._core.unrolling import Bounds

BOUNDS = Bounds(waveforms=Bounds(waveforms=1, readout=1, instructions=1))
SAMPLING_RATE = 5.89824
ADC_SAMPLING_RATE = 1.96608


class RFSOC(AWG):

    device: IcarusQRFSoC = None
    internal_channel_mapping: dict[str, int]
    bounds: str = "icarusq/bounds"

    def connect(self):
        host, port = self.address.split(":")
        self.device = IcarusQRFSoC(host, int(port))

    def sampling_rate(self) -> int:
        return SAMPLING_RATE

    def play(
        self,
        configs: dict[str, Config],
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[int, Result]:

        if len(sweepers) != 0:
            return self.recursive_sweep(configs, sequences, options, sweepers)

        results = {}
        for sequence in sequences:
            new_sequence = sequence.align_to_delays()
            qibolab_channel_waveform_map = self.generate_waveforms(
                new_sequence, configs
            )
            rfsoc_channel_waveform_map = {}
            acquisitions: dict[int, PulseId] = {}

            for channel, waveform in qibolab_channel_waveform_map.items():

                if isinstance(self.channels[channel], AcquisitionChannel):
                    continue
                elif isinstance(self.channels[channel], IqChannel):
                    waveform = waveform[:, 0] + waveform[:, 1]

                port = self.channels[channel].port
                waveform.resize(self.device.dac[port].max_samples)
                if port in rfsoc_channel_waveform_map:
                    rfsoc_channel_waveform_map[port] += waveform
                else:
                    rfsoc_channel_waveform_map[port] = waveform

            for channel, inputops in new_sequence.acquisitions:
                if inputops.kind == "acquisition":
                    raise RuntimeError(
                        "IcarusQ currently does not support acquisition events"
                    )

                channel_pulses = new_sequence.channel(channel)
                if channel_pulses[0].kind == "delay":
                    delay_adc = (
                        int(
                            np.ceil(channel_pulses[0].duration * ADC_SAMPLING_RATE / 8)
                            / 2
                        )
                        * 2
                    )
                    delay_dac = delay_adc * 1.5
                else:
                    delay_adc = 0
                    delay_dac = 0

                dac, adc = self.channels[channel].path.split(":")
                dac_port = int(dac)
                adc_port = int(adc)
                frequency = configs[channel].frequency
                waveform = self.generate_iq_waveform(
                    [inputops.probe],
                    SAMPLING_RATE,
                    self.device.dac[dac_port].max_samples,
                    frequency,
                )
                if dac_port in rfsoc_channel_waveform_map:
                    rfsoc_channel_waveform_map[dac_port] += waveform
                else:
                    rfsoc_channel_waveform_map[dac_port] = waveform
                rfsoc_acq_id = len(acquisitions)
                acquisitions[rfsoc_acq_id] = inputops.id
                self.device.program_qunit(
                    frequency, inputops.probe.duration * 1e-9, rfsoc_acq_id
                )

            self.device.dac[dac_port].delay = delay_dac
            self.device.adc[adc_port].delay = delay_adc

            payload = [
                (port, waveform, None)
                for port, waveform in rfsoc_channel_waveform_map.items()
            ]
            self.device.upload_waveform(payload)

            if options.acquisition_type is AcquisitionType.RAW:
                self.device.arm_adc([adc_port], options.nshots)
                temp = self.device.result()
                result = {acq_id: temp[adc_port] for acq_id in acquisitions.values()}

            else:
                if options.acquisition_type is AcquisitionType.INTEGRATION:
                    self.device.set_qunit_mode(0)
                elif options.acquisition_type is AcquisitionType.DISCRIMINATION:
                    self.device.set_qunit_mode(1)

                temp = self.device.start_qunit_acquisition(
                    options.nshots, list(acquisitions.values())
                )
                if options.acquisition_type is AcquisitionType.INTEGRATION:
                    result = {
                        acquisitions[rfsoc_acq_id]: np.dstack((i_array, q_array))[0]
                        for rfsoc_acq_id, (i_array, q_array) in temp.items()
                    }
                elif options.acquisition_type is AcquisitionType.DISCRIMINATION:
                    result = {
                        acq_id: temp[rfsoc_acq_id]
                        for rfsoc_acq_id, acq_id in acquisitions.items()
                    }

            if options.averaging_mode is not AveragingMode.SINGLESHOT:
                result = {
                    acq_id: np.average(array, axis=0)
                    for acq_id, array in result.items()
                }

            results.update(result)

        return results
