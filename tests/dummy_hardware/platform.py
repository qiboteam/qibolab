"""Dummy platform to test loading via ``Hardware`` object."""

from qibolab import Hardware
from qibolab._core.dummy.platform import create_dummy_hardware


def create() -> Hardware:
    return create_dummy_hardware()
