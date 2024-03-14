# import re
import socket

from qibo.config import log

# from datetime import datetime


class TemperatureController:
    """Bluefors temperature controller."""

    def __init__(self, ip_address: str, port: int = 8888):
        """Creation of the controller object.

        Args:
            ip_address (str): IP address of the board sending cryo temperature data.
            port (int): port of the board sending cryo temperature data.
        """
        self.ip_address = ip_address
        self.port = port

    def get_data(self) -> str:
        """Connect to the socket and get temperature data.

        Returns:
            message (str): socket message in this format:
                flange_name: {'temperature': <value(float)>, 'timestamp':<value(float)>}
        """
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.ip_address, self.port))
        log.info("Bluefors Temperature Controller Connected")

        # This message is a string that looks like a json
        # Shoud we convert it into a dictionary?
        message = client_socket.recv(1024).decode()
        return message

    def read_data(self):
        """Continously read data from the temperature controller."""
        while True:
            yield self.get_data()


# Example usage
if __name__ == "__main__":
    tc = TemperatureController("192.168.0.114", 8888)
    temperature_values = tc.read_data()
    for temperature_value in temperature_values:
        print(temperature_value)
