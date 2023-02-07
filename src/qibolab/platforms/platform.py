from qibolab.platforms.abstract import AbstractPlatform


class DesignPlatform(AbstractPlatform):
    """Platform that using an instrument design.

    This will maybe replace the ``AbstractPlatform`` object
    and work as a generic platform that works with an arbitrary
    ``InstrumentDesign``.
    """

    def __init__(self, name, design, runcard):
        super().__init__(name, runcard)
        self.design = design

    def connect(self):
        self.design.connect()
        self.is_connected = True

    def setup(self):
        self.design.setup(self.qubits, **self.settings["settings"])

    def start(self):
        self.design.start()

    def stop(self):
        self.design.stop()

    def disconnect(self):
        self.design.disconnect()
        self.is_connected = False

    def execute_pulse_sequence(self, sequence, nshots=1024, relaxation_time=None):
        if relaxation_time is None:
            relaxation_time = self.relaxation_time
        return self.design.play(self.qubits, sequence, nshots=nshots, relaxation_time=relaxation_time)

    def sweep(self, sequence, *sweepers, nshots=1024, relaxation_time=None, average=True):
        if relaxation_time is None:
            relaxation_time = self.relaxation_time
        return self.design.sweep(
            self.qubits, sequence, *sweepers, nshots=nshots, relaxation_time=relaxation_time, average=average
        )
