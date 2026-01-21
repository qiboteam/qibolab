from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from qibolab._core.components.channels import Channel
from qibolab._core.components.configs import (
    Config,
    DcConfig,
    IqMixerConfig,
    OscillatorConfig,
)
from qibolab._core.components.filters import (
    ExponentialFilter,
    FiniteImpulseResponseFilter,
)
from qibolab._core.serialize import Model

from ..identifiers import SlotId

__all__ = []


class PortAddress(Model):
    slot: SlotId
    ports: tuple[int, Optional[int]]
    input: bool = False

    @classmethod
    def from_path(cls, path: str):
        """Load address from :attr:`qibolab.Channel.path`."""
        els = path.split("/")
        assert len(els) == 2
        ports = els[1][1:].split("_")
        assert 1 <= len(ports) <= 2
        return cls(
            slot=int(els[0]),
            ports=(int(ports[0]), int(ports[1]) if len(ports) == 2 else None),
            input=els[1][0] == "i",
        )

    @property
    def direction(self) -> Literal["io", "out"]:
        """Signal flow direction.

        This is used for :meth:`local_address`, check its description for the in-depth
        meaning.

        .. note::

            While three directions are actually possible, there is no usage of pure
            acquisition channels in this driver (corresponding to the ``"in""``
            direction), and only support :class:`qibolab.Readout` instructions.

            For this reason, an input channel is always resolved to ``"io"``, implying
            that a same sequencer will be used to control both input (acquisitions) and
            output (probe pulses).
        """
        return "io" if self.input else "out"

    @property
    def local_address(self):
        """Physical address within the module.

        It will generate a string in the format ``<direction><channel>`` or
        ``<direction><I-channel>_<Q-channel>``.
        ``<direction>`` is ``in`` for a connection between an input and the acquisition
        path, ``out`` for a connection from the waveform generator to an output, or
        ``io`` to do both.
        The channels must be integer channel indices.
        Only one channel is present for a real mode operating sequencer; two channels
        are used for complex mode.

        .. note::

            Description adapted from
            https://docs.qblox.com/en/main/api_reference/cluster.html#qblox_instruments.Cluster.connect_sequencer
        """
        for port in self.ports:
            if port is not None:
                assert port > 0
        channels = (
            str(self.ports[0] - 1)
            if self.ports[1] is None
            else f"{self.ports[0] - 1}_{self.ports[1] - 1}"
        )
        return f"{self.direction}{channels}"


FilterConfig = Literal["bypassed", "enabled", "delay_comp"]


class PortConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    # DC offset
    offset: Optional[float] = None
    # LO parameters
    lo_en: Optional[bool] = None
    lo_freq: Optional[int] = None
    att: Optional[int] = None
    # mixer calibration
    offset_path0: Optional[float] = None
    offset_path1: Optional[float] = None
    # filters
    fir_config: Optional[FilterConfig] = None
    exp0_config: Optional[FilterConfig] = None
    exp1_config: Optional[FilterConfig] = None
    exp2_config: Optional[FilterConfig] = None
    exp3_config: Optional[FilterConfig] = None
    ## filter coefficients
    fir_coeffs: Optional[list[float]] = None
    exp0_time_constant: Optional[float] = None
    exp0_amplitude: Optional[float] = None
    exp1_time_constant: Optional[float] = None
    exp1_amplitude: Optional[float] = None
    exp2_time_constant: Optional[float] = None
    exp2_amplitude: Optional[float] = None
    exp3_time_constant: Optional[float] = None
    exp3_amplitude: Optional[float] = None

    @classmethod
    def build(
        cls,
        out: bool,
        in_: bool,
        channel: Channel,
        config: Config,
        lo: Optional[OscillatorConfig],
        mixer: Optional[IqMixerConfig],
    ) -> "PortConfig":
        n = PortAddress.from_path(channel.path).ports[0] - 1
        path = (f"out{n}_in{n}" if in_ else f"out{n}") if out else f"in{n}"
        port = cls(path=path)

        port.delay_compensation()
        if isinstance(config, DcConfig):
            port.offset_(config)
            port.filters(config)
        if lo is not None:
            port.lo(lo, in_)
        if mixer is not None:
            port.mixer(mixer)

        return port

    def offset_(self, dc: DcConfig) -> None:
        self.offset = dc.offset

    def lo(self, lo: OscillatorConfig, in_: bool) -> None:
        self.lo_en = True
        self.lo_freq = int(lo.frequency)
        self.att = int(lo.power) if not in_ else 0

    def mixer(self, lo: IqMixerConfig) -> None:
        self.offset_path0 = lo.offset_i
        self.offset_path1 = lo.offset_q

    def delay_compensation(self) -> None:
        self.fir_config = "delay_comp"
        for n in range(4):
            setattr(self, f"exp{n}_config", "delay_comp")

    def filters(self, dc: DcConfig) -> None:
        firs = [f for f in dc.filters if isinstance(f, FiniteImpulseResponseFilter)]
        assert len(firs) <= 1, "At most 1 FIR filter available"
        if len(firs) == 1:
            fir = firs[0]
            self.fir_config = "enabled"
            self.fir_coeffs = fir.coefficients

        exps = [f for f in dc.filters if isinstance(f, ExponentialFilter)]
        assert len(exps) <= 4, "At most 4 exponential filters available"
        for m, exp in enumerate(exps):
            setattr(self, f"exp{m}_config", "enabled")
            setattr(self, f"exp{m}_amplitude", exp.amplitude)
            setattr(self, f"exp{m}_time_constant", exp.tau)
