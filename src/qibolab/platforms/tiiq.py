class TIIq:

    def __init__(self):
        from qibolab.instruments import PulsarQRM, PulsarQCM, SGS100A
        self.qrm = PulsarQRM("qrm", '192.168.0.2')
        self.qcm = PulsarQCM("qcm", '192.168.0.3')
        self.LO_qrm = SGS100A("LO_qrm", "192.168.0.7")
        self.LO_qcm = SGS100A("LO_qcm", "192.168.0.101")
        self.software_averages = None

    def setup(self, settings):
        self.LO_qrm.setup(**settings.get("_LO_QRM_settings"))
        self.LO_qcm.setup(**settings.get("_LO_QCM_settings"))
        self.qrm.setup(**settings.get("QRM_settings"))
        self.qcm.setup(**settings.get("QCM_settings"))

    def stop(self):
        self.LO_qrm.off()
        self.LO_qcm.off()
        self.qrm.stop()
        self.qcm.stop()

    def execute(self, qrm_sequence, qcm_sequence, folder="./data"):
        waveforms, program = self.qrm.translate(qrm_sequence)
        self.qrm.upload(waveforms, program, folder)

        waveforms, program = self.qcm.translate(qcm_sequence)
        self.qcm.upload(waveforms, program, folder)

        self.qcm.play_sequence()
        # TODO: Find a better way to pass the frequency of readout pulse here
        acquisition_results = self.qrm.play_sequence_and_acquire(qrm_sequence.readout_pulse)
        return acquisition_results
