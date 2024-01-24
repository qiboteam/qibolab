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


def create_platform(name, path: Path = None) -> Platform:
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
        raise_error(ValueError, f"Platform {name} does not exist.")

    spec = importlib.util.spec_from_file_location("platform", platform / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if path is None:
        return module.create()
    return module.create(path)


def execute_qasm(circuit: str, platform, runcard=None, initial_state=None, nshots=1000):
    """Executes a QASM circuit.

    Args:
        circuit (str): the QASM circuit.
        platform (str): the platform where to execute the circuit.
        runcard (pathlib.Path): the path to the runcard used for the platform.
        initial_state (:class:`qibo.models.circuit.Circuit`): Circuit to prepare the initial state.
                If ``None`` the default ``|00...0>`` state is used.
        nshots (int): Number of shots to sample from the experiment.

    Returns:
        ``MeasurementOutcomes`` object containing the results acquired from the execution.
    """
    from qibolab.backends import QibolabBackend

    circuit = Circuit.from_qasm(circuit)
    return QibolabBackend(platform, runcard).execute_circuit(
        circuit, initial_state=initial_state, nshots=nshots
    )
