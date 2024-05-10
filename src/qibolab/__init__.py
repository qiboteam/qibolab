import importlib.metadata as im
import importlib.util
import os
from pathlib import Path

from qibo import Circuit
from qibo.config import raise_error

from qibolab.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab.platform import Platform
from qibolab.serialize import PLATFORM

__version__ = im.version(__package__)

PLATFORMS = "QIBOLAB_PLATFORMS"


def get_platforms_path():
    """Get path to repository containing the platforms.

    Path is specified using the environment variable QIBOLAB_PLATFORMS.
    """
    profiles = os.environ.get(PLATFORMS)
    if profiles is None or not os.path.exists(profiles):
        raise_error(RuntimeError, f"Profile directory {profiles} does not exist.")
    return Path(profiles)


def create_platform(name) -> Platform:
    """A platform for executing quantum algorithms.

    It consists of a quantum processor QPU and a set of controlling instruments.

    Args:
        name (str): name of the platform. Options are 'tiiq', 'qili' and 'icarusq'.
        path (pathlib.Path): path with platform serialization
    Returns:
        The plaform class.
    """
    if name == "dummy" or name == "dummy_couplers":
        from qibolab.dummy import create_dummy

        return create_dummy(with_couplers=name == "dummy_couplers")

    platform = get_platforms_path() / f"{name}"
    if not platform.exists():
        raise_error(
            ValueError,
            f"Platform {name} does not exist. Please pick one within the platforms found available {get_available_platforms()}.",
        )

    spec = importlib.util.spec_from_file_location("platform", platform / PLATFORM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create()


def execute_qasm(circuit: str, platform, initial_state=None, nshots=1000):
    """Executes a QASM circuit.

    Args:
        circuit (str): the QASM circuit.
        platform (str): the platform where to execute the circuit.
        initial_state (:class:`qibo.models.circuit.Circuit`): Circuit to prepare the initial state.
                If ``None`` the default ``|00...0>`` state is used.
        nshots (int): Number of shots to sample from the experiment.

    Returns:
        ``MeasurementOutcomes`` object containing the results acquired from the execution.
    """
    from qibolab.backends import QibolabBackend

    circuit = Circuit.from_qasm(circuit)
    return QibolabBackend(platform).execute_circuit(
        circuit, initial_state=initial_state, nshots=nshots
    )


def get_available_platforms() -> list[str]:
    """Returns the platforms found in the $QIBOLAB_PLATFORMS directory."""
    return [
        d.name
        for d in get_platforms_path().iterdir()
        if d.is_dir()
        and Path(f"{os.environ.get(PLATFORMS)}/{d.name}/platform.py") in d.iterdir()
    ]


class MetaBackend:
    """Meta-backend class which takes care of loading the qibolab backend."""

    @staticmethod
    def load(platform: str):
        """Loads the backend.

        Args:
            platform (str): Name of the platform to load.
        Returns:
            qibo.backends.abstract.Backend: The loaded backend.
        """
        from qibolab.backends import QibolabBackend

        return QibolabBackend(platform=platform)

    def list_available(self) -> dict:
        """Lists all the available qibolab platforms."""
        return {platform: True for platform in get_available_platforms()}
