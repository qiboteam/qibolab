import socket

import yaml
from pydantic import Field
from qibo.config import log

from qibolab.instruments.abstract import Instrument


class TemperatureController(Instrument):
    """Bluefors temperature controller.

    Example usage::

        if __name__ == "__main__":
            tc = TemperatureController("XLD1000_Temperature_Controller", "192.168.0.114", 8888)
            tc.connect()
            temperature_values = tc.read_data()
            for temperature_value in temperature_values:
                print(temperature_value)
    """

    address: str
    """IP address of the board sending cryo temperature data."""
    port: int = 8888
    """Port of the board sending cryo temperature data."""
    client_socket: socket.socket = Field(
        default_factory=lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    )
    _is_connected: bool = False

    def connect(self):
        """Connect to the socket."""
        if self._is_connected:
            return
        log.info(f"Bluefors connection. IP: {self.address} Port: {self.port}")
        self.client_socket.connect((self.address, self.port))
        self._is_connected = True
        log.info("Bluefors Temperature Controller Connected")

    def disconnect(self):
        """Disconnect from the socket."""
        if self._is_connected:
            self.client_socket.close()
            self._is_connected = False

    def get_data(self) -> dict[str, dict[str, float]]:
        """Connect to the socket and get temperature data.

        The typical message looks like this::

            flange_name: {'temperature':12.345678, 'timestamp':1234567890.123456}

        ``timestamp`` can be converted to datetime using ``datetime.fromtimestamp``.

        Returns:
            message (dict[str, dict[str, float]]): socket message in this format:
                {"flange_name": {'temperature': <value(float)>, 'timestamp':<value(float)>}}
        """
        return yaml.safe_load(self.client_socket.recv(1024).decode())

    def read_data(self):
        """Continously read data from the temperature controller."""
        while True:
            yield self.get_data()
