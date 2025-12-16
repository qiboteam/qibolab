import xarray as xr
from qblox_scheduler import HardwareAgent, Schedule
from qblox_scheduler.schedules.schedule import TimeableSchedule

from qibolab._core.components import Configs
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

SAMPLING_RATE = 1


class QBSchedulerController(Controller):
    """Controller object for Qblox-scheduler"""

    name: str
    bounds: str = "qblox/bounds"

    @property
    def sampling_rate(self) -> int:
        """Provide instrument's sampling rate."""
        return SAMPLING_RATE

    def _get_graph(self, cluster_name):
        graph = []
        for key, value in self.channels.items():
            qubit = key.split("/")[0]
            module, portstr = value.path.split("/")
            if portstr in ("o1", "i1"):
                port = 0
            if portstr == "o2":
                port = 1
            if portstr == "o3":
                port = 2
            if portstr == "o4":
                port = 3

            if key.endswith("probe"):
                out_type = "complex"
                port_name = "mw"
                iostr = "output"
            if key.endswith("acquisition"):
                out_type = "complex"
                port_name = "res"
                iostr = "input"
            if key.endswith("drive"):
                out_type = "complex"
                port_name = "mw"
                iostr = "output"
            if key.endswith("flux"):
                out_type = "real"
                port_name = "fl"
                iostr = "output"

            graph.append(
                (
                    f"{cluster_name}.module{module}.{out_type}_{iostr}_{port}",
                    f"q{qubit}:{port_name}",
                )
            )
        return graph

    def connect(self):
        """Connect and initialize the instrument."""

        cluster_name = (
            "cluster"  # we only support machines consisting of a single cluster
        )

        # https://docs.qblox.com/en/main/products/qblox_scheduler/user_guide/hardware_config.html
        hardware_cfg = {
            "hardware_description": {
                cluster_name: {
                    "instrument_type": "Cluster",
                    "ip": self.address,
                    "sequence_to_file": False,
                    "ref": "internal",
                }
            },
            "hardware_options": {},  # can be empty but has to exist
            "connectivity": {"graph": self._get_graph(cluster_name)},
        }

        self.agent = HardwareAgent(hardware_cfg)
        self.agent.connect_clusters()

    def disconnect(self):
        """Disconnect and reset the instrument."""
        # NOTE: HardwareAgent does not have a disconnect function

    def _timeable_schedule(
        self, pulse_sequence: PulseSequence
    ) -> Schedule | TimeableSchedule:
        ...
        return

    def _reshape_results(self, scheduler_result: xr.Dataset) -> dict:
        ...
        return

    def play(
        self,
        configs: Configs,
        sequences: list[PulseSequence],
        options: ExecutionParameters,
        sweepers: list[ParallelSweepers],
    ) -> dict[PulseId, Result]:
        """Play a pulse sequence and retrieve feedback.

        If :class:`qibolab.Sweeper` objects are passed as arguments, they are
        executed in real-time. If not possible, an error is raised.

        Returns a mapping with the id of the probe pulses used to acquired data.
        """

        results = {}
        for ps in sequences:
            # _PulseLike -> Schedule | TimeableSchedule
            sch = self._timeable_schedule(ps)
            schedule = self.agent.compile(sch)
            res = self.agent.run(schedule)
            # res: xarray
            # psres: dict[key: array]
            psres = self._reshape_results(res)

            results |= psres
        return results
