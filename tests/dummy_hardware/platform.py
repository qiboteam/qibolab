"""Dummy platform to test loading via ``Hardware`` object."""

from pathlib import Path

from qibolab import Hardware, initialize_parameters
from qibolab._core.dummy.platform import create_dummy_hardware
from qibolab._core.platform.platform import PARAMETERS


def create() -> Hardware:
    hardware = create_dummy_hardware()
    parameters = initialize_parameters(hardware=hardware)
    (Path(__file__).parent / PARAMETERS).write_text(
        parameters.model_dump_json(indent=4)
    )
    return hardware
