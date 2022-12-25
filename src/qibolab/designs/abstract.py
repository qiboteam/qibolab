from abc import ABC, abstractmethod


class AbstractInstrumentDesign(ABC):
    @abstractmethod
    def connect(self):
        """Connect to all instruments."""
        raise_error(NotImplementedError)

    @abstractmethod
    def setup(self, qubits):
        """Setup parameters of all instruments.

        Args:
            qubits (list): List of :class:`qibolab.platforms.utils.Qubit` objects
                that the instruments act on.
        """
        raise_error(NotImplementedError)

    @abstractmethod
    def start(self):
        """Start all instruments."""
        raise_error(NotImplementedError)

    @abstractmethod
    def stop(self):
        """Stop all instruments."""
        raise_error(NotImplementedError)

    @abstractmethod
    def disconnect(self):
        """Disconnect all instruments."""
        raise_error(NotImplementedError)

    # TODO: Add methods for sweeping

    @abstractmethod
    def play(self, qubits, sequence, nshots=1024):
        """Play an arbitrary pulse sequence and retrieve feedback.

        Args:
            qubits (list): List of :class:`qibolab.platforms.utils.Qubit` objects
                that the instruments act on.
            sequence (:class:`qibolab.pulses.PulseSequence`): Sequence of pulses to play.
            nshots (int): Number of hardware repetitions of the experiment.

        Returns:
            TODO: Decide a unified way to return results.
        """
        raise_error(NotImplementedError)
