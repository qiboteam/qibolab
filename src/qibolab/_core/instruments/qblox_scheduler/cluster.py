import xarray as xr
from qblox_scheduler import ClockResource, HardwareAgent, Schedule
from qblox_scheduler.operations import SimpleNumericalPulse
from qblox_scheduler.operations.acquisition_library import Trace
from qblox_scheduler.operations.pulse_library import Operation

from qibolab._core.components import Configs
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.identifier import Result
from qibolab._core.instruments.abstract import Controller
from qibolab._core.pulses import PulseId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers


class QBDelay(Operation):
    def __init__(self, duration: float, port: str, clock: str) -> None:
        super().__init__(name=self.__class__.__name__)
        self.data["pulse_info"] = {
            "wf_func": None,
            "t0": 0,
            "duration": duration,
            "clock": clock,
            "port": port,
        }


SAMPLING_RATE = 1


class QBSchedulerController(Controller):
    """Controller object for Qblox-scheduler"""

    name: str
    bounds: str = "qblox/bounds"

    @property
    def sampling_rate(self) -> int:
        """Provide instrument's sampling rate."""
        return SAMPLING_RATE

    def connect(self):
        """Connect and initialize the instrument."""
        # NOTE: We connect HardwareAgent at every play since we have to add
        # clocks with their frequencies, these are in `configs` so we can not do
        # it before play. We could initialize HardwareAgent without this
        # information and add the clocks when writing the `Sequence`, but then,
        # upon multiple calls to `play`, we end up writing the same clocks to an
        # exising HardwareAgent instance that alreay has them

    def disconnect(self):
        """Disconnect and reset the instrument."""
        # NOTE: HardwareAgent does not have a disconnect function

    def _get_graph(self, cluster_name, configs):
        graph = []
        modulation_frequencies = {}
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
                port_name = "res"
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

            if key.endswith("acquisition"):
                ...
            if key.endswith("flux"):
                # For the flux channel we don't set frequencies
                pass
            if key.endswith("drive") or key.endswith("probe"):
                lo_key = self.channels[key].lo
                lo_freq = configs[lo_key].frequency
                clock_freq = configs[key].frequency
                interm_freq = clock_freq - lo_freq
                modulation_frequencies[
                    f"{f'q{qubit}:{port_name}-q{qubit}.{key.split('/')[1]}'}"
                ] = {"lo_freq": lo_freq, "interm_freq": interm_freq}
        return graph, modulation_frequencies

    def _get_modulation_frequencies(self, configs) -> dict:
        # TODO: do I want to use this?
        return

    def _write_hardwar_cfg(self, configs):
        # we only support machines consisting of a single cluster
        cluster_name = "cluster"

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
            "hardware_options": {
                "modulation_frequencies": self._get_graph(cluster_name, configs)[1]
            },  # can be empty but has to exist
            "connectivity": {"graph": self._get_graph(cluster_name, configs)[0]},
        }
        return hardware_cfg

    def _handle_pulse(self, schedule, channel, pulse, configs):
        samples = pulse.i(self.sampling_rate) + 1j * pulse.q(self.sampling_rate)

        if channel.endswith("probe") or channel.endswith("drive"):
            port_name = "mw"
        if channel.endswith("flux"):
            port_name = "fl"

        port = f"q{channel.split('/')[0]}:{port_name}"

        clock = f"q{channel.split('/')[0]}:01"

        schedule.add(
            SimpleNumericalPulse(
                samples=samples,
                port=port,
                clock=clock,
            )
        )

    def _handle_acquisition(self, schedule, channel, acquisition, configs):
        qubit = channel.split("/")[0]
        freq = configs[f"{qubit}/probe"].frequency
        clock = ClockResource(name=f"q{qubit}.ro", freq=freq)
        schedule.add_resource(clock)
        schedule.add(
            Trace(
                duration=acquisition.duration * 1e-9,
                port=f"q{channel.split('/')[0]}:res",
                clock=clock.name,
            )
        )

    def _timeable_schedule(self, pulse_sequence: PulseSequence, configs) -> Schedule:
        schedule = Schedule("translated from PulseSequence")
        # basebandclock = ClockResource(name="cl0.baseband", freq=0.0)
        # schedule.add_resource(basebandclock)
        for channel, pulse in pulse_sequence:
            qubit = channel.split("/")[0]
            if pulse.kind == "readout":
                # readout consists of a probe and  acquisition
                self._handle_pulse(
                    schedule,
                    channel.replace("acquisition", "probe"),
                    pulse.probe,
                    configs,
                )
                self._handle_acquisition(schedule, channel, pulse.acquisition, configs)
            if pulse.kind == "pulse":
                self._handle_pulse(schedule, channel, pulse)
            if pulse.kind == "delay":
                duration = pulse.duration * 1e-9
                if channel.endswith("probe"):
                    port_name = "mw"
                if channel.endswith("acquisition"):
                    port_name = "res"
                if channel.endswith("drive"):
                    port_name = "mw"
                if channel.endswith("flux"):
                    port_name = "fl"
                schedule.add(
                    QBDelay(
                        duration, port=f"q{qubit}:{port_name}", clock=f"q{qubit}.01"
                    )
                )
            if pulse.kind == "acquisition":
                self._handle_acquisition(schedule, channel, pulse.acquisition, configs)
            if pulse.kind == "align":
                ...
            if pulse.kind == "virtualz":
                ...
        return schedule

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

        # TODO: add clocks (or only their frequencys) to config

        hardware_cfg = self._write_hardwar_cfg(configs)
        agent = HardwareAgent(hardware_cfg)
        agent.connect_clusters()

        results = {}
        for ps in sequences:
            # PulseSequence -> Schedule | TimeableSchedule
            timeable_schedule = self._timeable_schedule(ps, configs)
            schedule = agent.compile(timeable_schedule)
            res = agent.run(schedule)
            # res: xarray
            # psres: dict[key: array]
            psres = self._reshape_results(res)

            results |= psres
        return results
