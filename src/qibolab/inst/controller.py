import numpy as np
from qcodes.instrument_drivers.AlazarTech import ATS
from qibolab.inst import AWG5204, QuicSyn, MCAttenuator, AlazarTech_ATS9371

trigger_volts = 1

class InstrumentController(ATS.AcquisitionController):
    def __init__(self, name="alz_cont", alazar_name="Alazar1", **kwargs):

        self.awg = AWG5204()
        self.lo = QuicSyn("ASRL3::INSTR")
        self.qubit_attenuator = MCAttenuator("192.168.0.9:90")
        self.readout_attenuator = MCAttenuator("192.168.0.10:100")
        self.adc = AlazarTech_ATS9371(alazar_name)
        self.setup_alazar()
        self.acquisitionkwargs = {}
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        # TODO(damazter) (S) this is not very general:
        self.number_of_channels = 2
        self.buffer = None
        super().__init__(name, alazar_name, **kwargs)
        self.add_parameter("acquisition", get_cmd=self.do_acquisition)


    def setup_alazar(self):
        input_range_volts = 2.5
        trigger_level_code = int(128 + 127 * trigger_volts / input_range_volts)
        with self.adc.syncing():
            self.adc.clock_source("EXTERNAL_CLOCK_10MHz_REF")
            #self.adc.clock_source("INTERNAL_CLOCK")
            self.adc.external_sample_rate(1_000_000_000)
            #self.adc.sample_rate(1_000_000_000)
            self.adc.clock_edge("CLOCK_EDGE_RISING")
            self.adc.decimation(1)
            self.adc.coupling1('DC')
            self.adc.coupling2('DC')
            self.adc.channel_range1(.02)
            #self.adc.channel_range2(.4)
            self.adc.channel_range2(.02)
            self.adc.impedance1(50)
            self.adc.impedance2(50)
            self.adc.bwlimit1("DISABLED")
            self.adc.bwlimit2("DISABLED")
            self.adc.trigger_operation('TRIG_ENGINE_OP_J')
            self.adc.trigger_engine1('TRIG_ENGINE_J')
            self.adc.trigger_source1('EXTERNAL')
            self.adc.trigger_slope1('TRIG_SLOPE_POSITIVE')
            self.adc.trigger_level1(trigger_level_code)
            self.adc.trigger_engine2('TRIG_ENGINE_K')
            self.adc.trigger_source2('DISABLE')
            self.adc.trigger_slope2('TRIG_SLOPE_POSITIVE')
            self.adc.trigger_level2(128)
            self.adc.external_trigger_coupling('DC')
            self.adc.external_trigger_range('ETR_2V5')
            self.adc.trigger_delay(0)
            #self.aux_io_mode('NONE') # AUX_IN_TRIGGER_ENABLE for seq mode on
            #self.aux_io_param('NONE') # TRIG_SLOPE_POSITIVE for seq mode on
            self.adc.timeout_ticks(0)

    def update_acquisitionkwargs(self, **kwargs):
        """
        This method must be used to update the kwargs used for the acquisition
        with the alazar_driver.acquire
        :param kwargs:
        :return:
        """
        self.acquisitionkwargs.update(**kwargs)

    def update_adc(self, samples, averaging):
        self.update_acquisitionkwargs(mode='NPT',
                                      samples_per_record=samples,
                                      records_per_buffer=10,
                                      buffers_per_acquisition=int(averaging / 10),
                                      interleave_samples='DISABLED',
                                      allocated_buffers=100,
                                      buffer_timeout=100000)


    def do_acquisition(self):
        """
        this method performs an acquisition, which is the get_cmd for the
        acquisiion parameter of this instrument
        :return:
        """
        value = self._get_alazar().acquire(acquisition_controller=self,
                                           **self.acquisitionkwargs)
        return value


    def pre_start_capture(self):
        """
        See AcquisitionController
        :return:
        """

        self.samples_per_record = self.adc.samples_per_record.get()
        self.records_per_buffer = self.adc.records_per_buffer.get()
        self.buffers_per_acquisition = self.adc.buffers_per_acquisition.get()
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels)


    def pre_acquire(self):
        """
        See AcquisitionController
        :return:
        """
        # this could be used to start an Arbitrary Waveform Generator, etc...
        # using this method ensures that the contents are executed AFTER the
        # Alazar card starts listening for a trigger pulse
        self.awg.trigger()

    def handle_buffer(self, data, buffer_number=None):
        """
        See AcquisitionController
        :return:
        """
        self.buffer += data

    def post_acquire(self):
        """
        See AcquisitionController
        :return:
        """
        return self.buffer, self.buffers_per_acquisition, self.records_per_buffer, self.samples_per_record

    def stop(self):
        self.awg.stop()
        self.lo.stop()

    def close(self):
        self.awg.close()
        self.lo.close()
        self.adc.close()
        super().close()
