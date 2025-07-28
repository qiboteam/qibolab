from typing import Optional

from pydantic import Field
from spirack import S4g_module, SPI_rack

from ..components import DcChannel, DcConfig
from ..identifier import ChannelId
from ..parameters import ComponentId
from ..serialize import Model
from .abstract import Instrument

__all__ = ["Spi"]


def channel_to_dac(channel: DcChannel) -> tuple[int, int]:
    """Read module and DAC numbers from channel object.

    Assumes that ``channel.path`` has the following format:
    ``{module_number}/{dac_number}``
    """
    module, dac = channel.path.split("/")
    return int(module), int(dac)


class S4g(Model):
    """Driver for S4g modules of the SPI rack.

    Uses https://qtwork.tudelft.nl/~mtiggelman/modules/i-source/s4g.html
    """

    module: Optional[S4g_module] = None
    currents: dict[int, float] = Field(default_factory=dict)
    """Currents cache on software (maintained even when we are not connected to the actual device)."""

    def connect(self, spi: SPI_rack, number: int, reset_currents: bool = False):
        if self.module is None:
            self.module = S4g_module(spi, number, reset_currents=reset_currents)
        self.upload()

    def upload(self):
        """Upload currents from our cache to the instrument."""
        for dac, current in self.currents.items():
            self.set_current(self.module, dac, current)

    def set_current(self, dac: int, current: float):
        """Update current value in cache and, if connected, upload to the instrument."""
        self.currents[dac] = current
        if self.module is not None:
            self.module.set_current(dac, current)


class Spi(Instrument):
    """Driver for SPI rack.

    Uses https://qtwork.tudelft.nl/~mtiggelman/spi-rack.html

    Currently only supports S4g modules.
    """

    channels: dict[ChannelId, DcChannel] = Field(default_factory=dict)
    close_currents: bool = False
    baud: int = 9600
    timeout: int = 1
    modules: dict[int, S4g_module] = Field(default_factory=dict)

    def __post_init__(self):
        self.modules = {
            i: S4g() for i in {channel_to_dac(ch)[0] for ch in self.channels.values()}
        }

    def connect(self):
        """Connect to the instrument."""
        spi = SPI_rack(port=self.address, baud=self.baud, timeout=self.timeout)
        spi.unlock()
        for nr, module in self.modules.items():
            module.connect(spi, nr)

    def disconnect(self):
        if self.close_currents:
            for module in self.modules.values():
                for dac in module.currents:
                    module.set_current(dac, 0)

    def channel_module(self, channel_name: ChannelId) -> Optional[S4g_module]:
        """Get ``spirack.S4g_module`` used on a particular channel."""
        return self.modules[channel_to_dac(self.channels[channel_name])[0]].module

    def setup(self, configs: dict[ComponentId, DcConfig]):
        """Update currents based on channel configs.

        If the instrument is connected the value is automatically uploaded to the instrument.
        Otherwise the value is cached and will be uploaded when connection is established.
        """
        for channel_name, config in configs.items():
            module, dac = channel_to_dac(self.channels[channel_name])
            if module in self.modules:
                self.modules[module].set_current(dac, config.offset)
