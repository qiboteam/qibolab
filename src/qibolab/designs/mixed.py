from dataclasses import dataclass

from qibo.config import log, raise_error

from qibolab.designs.basic import BasicInstrumentDesign


@dataclass
class MixedInstrumentDesign(BasicInstrumentDesign):
    """Mixer based instrument design that uses a controller and local oscillators.

    Attributes:
        controller (:class:`qibolab.instruments.abstract.AbstractInstrument`): Instrument used for sending pulses and retrieving feedback.
        is_connected (bool): Boolean that shows whether instruments are connected.
        local_oscillators (list): List of local oscillator instrument objects.
    """

    local_oscillators: list

    def connect(self):
        if not self.is_connected:
            for lo in self.local_oscillators:
                try:
                    log.info(f"Connecting to instrument {lo}.")
                    lo.connect()
                except Exception as exception:
                    raise_error(
                        RuntimeError,
                        f"Cannot establish connection to {lo} instruments. Error captured: '{exception}'",
                    )
        super().connect()

    def setup(self, qubits, *args, **kwargs):
        for qubit in qubits.values():
            # setup local oscillators
            for channel in qubit.channels:
                if channel.local_oscillator is not None:
                    lo = channel.local_oscillator
                    if lo.is_connected:
                        lo.setup(frequency=channel.lo_frequency, power=channel.lo_power)
                    else:
                        log.warn(f"There is no connection to {lo}. Frequencies were not set.")
        super().setup(qubits, *args, **kwargs)

    def start(self):
        if self.is_connected:
            for lo in self.local_oscillators:
                lo.start()
        super().start()

    def stop(self):
        if self.is_connected:
            for lo in self.local_oscillators:
                lo.stop()
        super().stop()

    def disconnect(self):
        if self.is_connected:
            for lo in self.local_oscillators:
                lo.disconnect()
        super().disconnect()
