# let's suppose that there is already avaiable a base driver for connection
# and control of the device
from proprietary_instruments import BiaserDriver

from qibolab.instruments.abstract import Instrument


class Biaser(Instrument):
    """An instrument that delivers constant biases."""

    def __init__(self, name, address, min_value=-1, max_value=1):
        super().__init__(name, address)
        self.max_value: float = (
            max_value  # attribute example, maximum value of voltage allowed
        )
        self.min_value: float = (
            min_value  # attribute example, minimum value of voltage allowed
        )
        self.bias: float = 0

        self.device = BiaserDriver(address)

    def connect(self):
        """Check if a connection is avaiable."""
        if not self.device.is_connected:
            raise ConnectionError("Biaser not connected")

    def disconnect(self):
        """Method not used."""

    # FIXME:: *args, **kwargs are not passed on
    def setup(self):
        """Set biaser parameters."""
        self.device.set_range(self.min_value, self.max_value)

    def start(self):
        """Start biasing."""
        self.device.on(bias)

    def stop(self):
        """Stop biasing."""
        self.device.off(bias)
