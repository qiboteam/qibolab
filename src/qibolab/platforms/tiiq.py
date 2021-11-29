import json


class TIIq:

    def __init__(self):
        self.qrm = Pulsar_QRM("qrm", '192.168.0.2')
        self.qcm = Pulsar_QCM("qcm", '192.168.0.3')

        from qibolab.instruments import SGS100A
        self.LO_qrm = SGS100A("LO_qrm", "192.168.0.7"")
        self.LO_qcm = SGS100A("LO_qcm", "192.168.0.101")

        self.software_averages = None

    def setup(self, filename):
        with open(filename, "r") as file:
            settings = json.load(file)

        self.LO_qrm.setup(**settings.get("_LO_QRM_settings"))
        self.LO_qcm.setup(**settings.get("_LO_QCM_settings"))

        self.qrm.setup(self.QRM_settings)
        self.qcm.setup(self.QCM_settings)

    def stop(self):
        self.LO_qrm.off()
        self.LO_qcm.off()
        self.qrm.stop()
        self.qcm.stop()
