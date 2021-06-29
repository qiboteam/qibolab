import pyvisa
from typing import Union

rm = pyvisa.ResourceManager()

class Instrument:
    def __init__(self, address: str, timeout: int = 10000) -> None:
        self._visa_handle = rm.open_resource(address, timeout=timeout)

    def write(self, msg: Union[bytes, str]) -> None:
        self._visa_handle.write(msg)

    def query(self, msg: Union[bytes, str]) -> str:
        return self._visa_handle.query(msg)

    def read(self) -> str:
        return self._visa_handle.read()

    def close(self) -> None:
        self._visa_handle.close()
