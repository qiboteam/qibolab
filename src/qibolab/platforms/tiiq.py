class TIIq:

    def __init__(self, settings):
        from qibolab.instruments import PulsarQRM, PulsarQCM, SGS100A
        self.qrm = PulsarQRM(**settings.get("_QRM_init_settings"))
        self.qcm = PulsarQCM(**settings.get("_QCM_init_settings"))
        self.LO_qrm = SGS100A(**settings.get("_LO_QRM_init_settings"))
        self.LO_qcm = SGS100A(**settings.get("_LO_QCM_init_settings"))

        self.data_folder = None
        self.hardware_avg = None
        self.sampling_rate = None
        self.software_averages = None
        self.repetition_duration = None
        self.setup_platform(settings.get("_settings"))

    def setup(self, settings):
        self.qrm.setup(**settings.get("_QRM_settings"))
        self.qcm.setup(**settings.get("_QCM_settings"))
        self.LO_qrm.setup(**settings.get("_LO_QRM_settings"))
        self.LO_qcm.setup(**settings.get("_LO_QCM_settings"))

    def setup_platform(self, settings):
        self.data_folder = settings.get("data_folder", self.data_folder)
        self.hardware_avg = settings.get("hardware_avg", self.hardware_avg)
        self.sampling_rate = settings.get("sampling_rate", self.sampling_rate)
        self.software_averages = settings.get("software_averages", self.software_averages)
        self.repetition_duration = settings.get("repetition_duration", self.repetition_duration)

    def stop(self):
        self.LO_qrm.off()
        self.LO_qcm.off()
        self.qrm.stop()
        self.qcm.stop()

    def __del__(self):
        self.LO_qrm.close()
        self.LO_qcm.close()
        self.qrm.close()
        self.qcm.close()

    def execute(self, qrm_sequence, qcm_sequence):
        waveforms, program = self.qrm.translate(qrm_sequence)
        self.qrm.upload(waveforms, program, self.data_folder)

        waveforms, program = self.qcm.translate(qcm_sequence)
        self.qcm.upload(waveforms, program, self.data_folder)

        self.qcm.play_sequence()
        # TODO: Find a better way to pass the frequency of readout pulse here
        acquisition_results = self.qrm.play_sequence_and_acquire(qrm_sequence.readout_pulse)
        return acquisition_results
