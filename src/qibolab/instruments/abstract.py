# -*- coding: utf-8 -*-
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from qibolab.paths import user_folder


class AbstractInstrument(ABC):
    """
    Parent class for all the instruments connected via TCPIP.

    Args:
        name (str): Instrument name.
        address (str): Instrument network address.
    """

    def __init__(self, name, address):
        self.name = name
        self.address = address
        self.is_connected = False
        self.signature = f"{type(self).__name__}@{address}"
        self.device = None
        # create local storage folder
        instruments_data_folder = user_folder / "instruments" / "data"
        instruments_data_folder.mkdir(parents=True, exist_ok=True)
        # create temporary directory
        if "QIBOLAB_KEEP_DATA" in os.environ:
            self.tmp_folder = tempfile.mkdtemp(dir=instruments_data_folder)
        else:
            self.tmp_folder = tempfile.TemporaryDirectory(dir=instruments_data_folder)
        self.data_folder = Path(self.tmp_folder.name)

    @abstractmethod
    def connect(self):
        raise NotImplementedError

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    @abstractmethod
    def disconnect(self):
        raise NotImplementedError


class InstrumentException(Exception):
    def __init__(self, instrument: AbstractInstrument, message: str):
        header = f"InstrumentException with {instrument.signature}"
        full_msg = header + ": " + message
        super().__init__(full_msg)
