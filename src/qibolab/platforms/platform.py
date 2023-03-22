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

    def execute_pulse_sequence(
        self, sequence, nshots=1024, relaxation_time=None, fast_reset=False, sim_time=10e-6, acquisition_type=None
    ):
        if relaxation_time is None:
            relaxation_time = self.relaxation_time
        if fast_reset is True:
            fast_reset = {}
            for qubit in self.qubits.values():
                if "c" in str(qubit.name):
                    pass
                else:
                    fast_reset[qubit.name] = self.create_RX_pulse(qubit=qubit.name, start=0)
        return self.design.play(
            self.qubits,
            sequence,
            nshots=nshots,
            relaxation_time=relaxation_time,
            fast_reset=fast_reset,
            sim_time=sim_time,
            acquisition_type=acquisition_type,
        )

    def sweep(self, sequence, *sweepers, nshots=1024, relaxation_time=None, average=True, sim_time=2e-6):
        if relaxation_time is None:
            relaxation_time = self.relaxation_time
        return self.design.sweep(
            self.qubits,
            sequence,
            *sweepers,
            nshots=nshots,
            relaxation_time=relaxation_time,
            average=average,
            sim_time=sim_time,
        )
