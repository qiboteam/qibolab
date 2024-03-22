import json
import re
import socket
from datetime import datetime

from qibo.config import log

from qibolab.instruments.abstract import Instrument


class TemperatureController(Instrument):
    """Bluefors temperature controller.

    ```
    # Example usage
    if __name__ == "__main__":
        tc = TemperatureController("XLD1000_Temperature_Controller", "192.168.0.114", 8888)
        tc.connect()
        temperature_values = tc.read_data()
        for temperature_value in temperature_values:
            print(temperature_value)
    ```
    """

    def __init__(self, name: str, address: str, port: int = 8888):
        """Creation of the controller object.

        Args:
            name (str): name of the instrument.
            address (str): IP address of the board sending cryo temperature data.
            port (int): port of the board sending cryo temperature data.
        """
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        super().__init__(name, address)

    def connect(self):
        """Connect to the socket."""
        if self.is_connected:
            return
        log.info(f"Bluefors connection. IP: {self.address} Port: {self.port}")
        self.client_socket.connect((self.address, self.port))
        self.is_connected = True
        log.info("Bluefors Temperature Controller Connected")

    def disconnect(self):
        """Disconnect from the socket."""
        if self.is_connected:
            self.client_socket.close()
            self.is_connected = False

    def setup(self):
        """Required by parent class, but not used here."""
        pass

    @staticmethod
    def convert_to_json(message: str) -> dict[str, dict[str, float]]:
        """Convert the received socket message into a dictionary.

        The typical message looks like this:
            flange_name: {'temperature':12.345678, 'timestamp':1234567890.123456}
        Args:
            message (str): messaged received from the socket.
        Returns:
            dictionary_message (dict[str, dict[str, float]]):
                message converted into python dictionary.
        """
        message = "\n".join(
            [re.sub("^([^':]+)", r"'\g<1>'", m) for m in message.split("\n")]
        )
        message = re.sub("'", '"', message)
        message = ",".join(message.split("\n"))
        dictionary_message = json.loads("{" + message + "}")
        for flange_values in dictionary_message.values():
            flange_values["time"] = datetime.fromtimestamp(flange_values["timestamp"])
        return dictionary_message

    def get_data(self) -> dict[str, dict[str, float]]:
        """Connect to the socket and get temperature data.

        Returns:
            message (dict[str, dict[str, float]]): socket message in this format:
                {"flange_name": {'temperature': <value(float)>, 'timestamp':<value(float)>}}
        """
        message = self.convert_to_json(self.client_socket.recv(1024).decode())
        return message

    def read_data(self):
        """Continously read data from the temperature controller."""
        while True:
            yield self.get_data()
