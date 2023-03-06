from dataclasses import dataclass, field
from typing import List

from qibo.config import log, raise_error

from qibolab.designs.channels import ChannelMap
from qibolab.instruments.abstract import AbstractInstrument


@dataclass
class InstrumentDesign:
    """Instrument design that uses a single controller.

    Attributes:
        instruments (list): List of :class:`qibolab.instruments.abstract.AbstractInstrument` objects.
        channels (:class:`qibolab.designs.ChannelMap`): Channel map describing the connections in the lab.
        _is_connected (bool): Boolean that shows whether instruments are connected.
    """

    instruments: List[AbstractInstrument]
    channels: ChannelMap = field(default_factory=ChannelMap)
    _is_connected: bool = field(default=False, init=False)

    def connect(self):
        """Connect to all instruments."""
        if not self._is_connected:
            for instrument in self.instruments:
                try:
                    log.info(f"Connecting to instrument {instrument}.")
                    instrument.connect()
                except Exception as exception:
                    raise_error(
                        RuntimeError,
                        f"Cannot establish connection to {instrument} instruments. Error captured: '{exception}'",
                    )
        self._is_connected = True

    def setup(self):
        """Load settings to instruments."""
        for instrument in self.instruments:
            instrument.setup()

    def start(self):
        """Start all instruments."""
        if self._is_connected:
            for instrument in self.instruments:
                instrument.start()

    def stop(self):
        """Stop all instruments."""
        if self._is_connected:
            for instrument in self.instruments:
                instrument.stop()

    def disconnect(self):
        """Disconnect all instruments."""
        if self._is_connected:
            for instrument in self.instruments:
                instrument.disconnect()
        self._is_connected = False

    def play(self, *args, **kwargs):
        """Play a pulse sequence and retrieve feedback."""
        result = {}
        for instrument in self.instruments:
            new_result = instrument.play(*args, **kwargs)
            if isinstance(new_result, dict):
                result.update(new_result)
            elif new_result is not None:
                # currently the result of QMSim is not a dict
                result = new_result
        return result

    def sweep(self, *args, **kwargs):
        """Play a pulse sequence while sweeping one or more parameters."""
        result = {}
        for instrument in self.instruments:
            new_result = instrument.sweep(*args, **kwargs)
            if isinstance(new_result, dict):
                result.update(new_result)
            elif new_result is not None:
                # currently the result of QMSim is not a dict
                result = new_result
        return result
