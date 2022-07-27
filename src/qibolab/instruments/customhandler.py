import sys
sys.path.append("/opt/hostedtoolcache/Python/3.8.13/x64/lib")
import usb

class DeviceHandle(usb.DeviceHandle):

    """Version of the usb.DeviceHandle that will be used to initialise and exit the board"""

    def __init__(self, dev):
        """Inherit everything from the super class."""
        super(DeviceHandle, self).__init__(dev)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
