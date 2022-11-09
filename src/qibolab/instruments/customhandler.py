import os
import sys

import numpy as np
import usb


class DeviceHandle(usb.DeviceHandle):

    """Version of the usb.DeviceHandle that will be used to initialise and exit the boards connected by the USB. Currently it is used for AD9959 only."""

    def __init__(self, dev):
        """Inherit everything from the super class."""
        super().__init__(dev)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
