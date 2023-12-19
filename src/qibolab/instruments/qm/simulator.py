from dataclasses import dataclass

from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from qualang_tools.simulator_tools import create_simulator_controller_connections

from .driver import QMOPX


@dataclass
class QMSim(QMOPX):
    """Instrument for using the Quantum Machines (QM) OPX simulator.

    Args:
        address (str): Address and port of the simulator.
        simulation_duration (int): Duration for the simulation in ns.
        cloud (bool): If ``True`` the QM cloud simulator is used which does not
            require access to physical instruments. This assumes that a proper
            cloud address has been given.
            If ``False`` the simulator built-in the instruments is used.
            This requires connection to instruments.
            Default is ``False``.
    """

    simulation_duration: int = 1000
    cloud: bool = False

    def __post_init__(self):
        """Convert simulation duration from ns to clock cycles."""
        self.simulation_duration //= 4
        self.settings = None

    def connect(self):
        host, port = self.address.split(":")
        if self.cloud:
            from qm.simulate.credentials import create_credentials

            self.manager = QuantumMachinesManager(
                host, int(port), credentials=create_credentials()
            )
        else:
            self.manager = QuantumMachinesManager(host, int(port))

    @staticmethod
    def fetch_results(result, ro_pulses):
        return result

    def execute_program(self, program):
        ncontrollers = len(self.config.controllers)
        controller_connections = create_simulator_controller_connections(ncontrollers)
        simulation_config = SimulationConfig(
            duration=self.simulation_duration,
            controller_connections=controller_connections,
        )
        return self.manager.simulate(self.config.__dict__, program, simulation_config)
