import re
import socket
import threading
from datetime import datetime

from qibo.config import log


class TemperatureController:
    def __init__(self, IP_address):
        self.IP_address = IP_address
        self._thread = threading.Thread(target=self.get_data)
        self._thread.daemon = (
            True  # Daemonize the thread so it terminates when the main program exits
        )
        self._is_running = False
        self._property = None
        self._thread.start()

    def get_data(self):

        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the server
        client_socket.connect((self.IP_address, 8888))
        log.info("Bluefors Temperature Controller Connected")

        self.data = {}
        while True:
            try:
                # Receive data from the server
                message = client_socket.recv(1024).decode()
                # print(message)

                # Define regular expression pattern to match the identifier, parameters, and values
                pattern = r"([\w-]+): {'(\w+)':([\d\.]+), '(\w+)':([\d\.]+)}"

                # Find all matches using regex
                matches = re.findall(pattern, message)

                # Extract identifier, parameters, and values
                flange, _, temperature, _, timestamp = matches[0]

                # Convert timestamp to datetime object
                date_time = datetime.fromtimestamp(float(timestamp))

                # Format datetime object as a string
                formatted_date = date_time.strftime("%Y-%m-%d %H:%M:%S")

                # Create a dictionary with the extracted information
                self.data[flange] = {
                    "temperature": float(temperature),
                    "timestamp": float(timestamp),
                    "datetime": formatted_date,
                }
                # print(data)
            except:
                pass


# Example usage
if __name__ == "__main__":
    import time

    tc = TemperatureController("192.168.0.114")
    while True:
        time.sleep(10)
        print("latest data")
        print(tc.data)
