import sys
#sys.path.append("/opt/hostedtoolcache/Python/3.8.13/x64/lib")
import numpy as np
import os
print('this is the usb version')
usb_path = np.__file__.replace("numpy", "usb")
assert os.path.isfile(usb_path)
#init-hook="from usb.config import find_pylintrc; import os, sys; sys.path.append(os.path.dirname(find_pylintrc()))"
import usb #pyLint: disable=E401

print(usb.__file__)
print(np.__file__)

class DeviceHandle(usb.DeviceHandle):

    """Version of the usb.DeviceHandle that will be used to initialise and exit the board"""

    def __init__(self, dev):
        """Inherit everything from the super class."""
        super(DeviceHandle, self).__init__(dev)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
