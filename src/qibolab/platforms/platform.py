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

    def is_connected(self):
        return self.design.is_connected

    def connect(self):
        self.design.connect()
        self.is_connected = True

    def setup(self):
        self.design.setup(self.qubits, self.channels, **self.options)

    def start(self):
        self.design.start()

    def stop(self):
        self.design.stop()

    def disconnect(self):
        self.design.disconnect()
        self.is_connected = False

    def execute_pulse_sequence(self, sequence, nshots=1024):
        """Play an arbitrary pulse sequence and retrieve feedback.

        Args:
            sequence (:class:`qibolab.pulses.PulseSequence`): Sequence of pulses to play.
            nshots (int): Number of hardware repetitions of the execution.

        Returns:
            TODO: Decide a unified way to return results.
        """
        return self.design.play(self.qubits, sequence, nshots)

    def sweep(self, sequence, *sweepers, nshots=1024, average=True):
        return self.design.sweep(self.qubits, sequence, *sweepers, nshots=nshots, average=average)
