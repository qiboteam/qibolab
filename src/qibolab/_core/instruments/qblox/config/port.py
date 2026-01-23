from itertools import groupby
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from qibolab._core.components.channels import AcquisitionChannel, Channel, IqChannel
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


QCM_SWEEP_TO_OFFSET = 2.5 / np.sqrt(2)
"""Conversion factor between swept value and configuration.

There are two different ways to add an offset to the waveform played by the QCM module:

- digitally summing an offset, which could be controlled both in real-time and by
  conifgurations
- adding an offset directly to the outcoming signal


Since the QCM supplies outputs at 5 Vpp (`documented as +/-2.5 V
<https://docs.qblox.com/en/main/products/architecture/modules/qcm.html#specifications>`_),
a conversion is neeeded, because the first option will be defined in the interval (-1,
1) in the parameters (internally mapping the floats on a suitable integers range), while
the second is directly expressed in Volt.
Hence, the conversion factor of ``2.5``.

However, these two ways are not equivalent, especially because of the NCO and LO mixing
process.
Indeed, the first one is happening upstream to the mixing process, and the second
downstream.
Since we are sweeping only one of the two components of the signal (the in-phase,
I), it will result multiplied by a sine-wave, which reduces its root mean square (RMS)
power by a factor of `sqrt(2)`. Which is then accounted for in the conversion range.

https://docs.qblox.com/en/main/products/architecture/sequencers/control.html#arbitrary-waveform-generator-awg
https://docs.qblox.com/en/main/products/architecture/modules/qcm.html#block-diagram

Notice that sweeping both of the components is also viable. But even without any flux
pulse, the sum of sine and cosine with maximal amplitude will saturate the power supply,
eventually clipping the signal and reducing the power range.
"""


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

    path: str = Field(exclude=True)
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
        channel: Channel,
        config: Config,
        in_: bool,
        out: bool,
        lo: Optional[OscillatorConfig],
        mixer: Optional[IqMixerConfig],
    ) -> "PortConfig":
        """Create port configuration for the desired channel.

        Multiple channels can share the same port, in which case they should set
        consistent configurations.
        """
        n = PortAddress.from_path(channel.path).ports[0] - 1

        # the port configureation should be used either for input or output - or "both",
        # which is possible on QRM modules
        assert in_ or out
        only_out = out and not in_
        in_out = in_ and out
        acq = isinstance(channel, AcquisitionChannel)
        drive = isinstance(channel, IqChannel)

        path = f"in{n}" if not out else (f"out{n}" if not in_ else f"out{n}_in{n}")
        port = cls(path=path)

        # delay compensation is applied by default on all output ports - time-of-flight
        # will be adjusted separately for input
        if only_out:
            port.delay_compensation()

        # DC channels are configured for static offsets and pre-distortions
        if isinstance(config, DcConfig):
            if only_out:
                port.offset_(config)
                port.filters(config)
        else:
            # on QRM-RF, we do not configure separately the LO for the input port, since
            # it is shared with the output one
            if lo is not None:
                # acquisition LO are then configured only for in-out, since related to
                # both, while drive channels are only configured for output
                if (acq and in_out) or (drive and only_out):
                    port.lo(lo)
                # the attenuation is only configured for individual physical ports,
                # since always separated (also on QRM)
                # of course, input attenuation is only available on QRM
                if (acq and not in_out) or only_out:
                    # TODO: input attenuation unused, but available in Qblox
                    port.att_(lo if not in_ else lo.model_copy(update={"power": 0.0}))

            # but we do have separate mixers for input and output (on QRM, and only out
            # on QCM)
            # TODO: input mixers unused, but available in Qblox
            # at the moment, it would share Qibolab configurations with the output one,
            # but there is no reason why the attenuation should be the same
            # we would need a separate `AcquisitionChannel.mixer` entry
            if mixer is not None and ((acq and only_out) or (drive and only_out)):
                port.mixer(mixer)

        return port

    def offset_(self, dc: DcConfig) -> None:
        self.offset = dc.offset * QCM_SWEEP_TO_OFFSET

    def lo(self, lo: OscillatorConfig) -> None:
        self.lo_en = True
        self.lo_freq = int(lo.frequency)

    def att_(self, lo: OscillatorConfig) -> None:
        self.att = int(lo.power)

    def mixer(self, mixer: IqMixerConfig) -> None:
        self.offset_path0 = mixer.offset_i
        self.offset_path1 = mixer.offset_q

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


StrDict = dict[str, Any]


def groupitems(items: list[tuple[str, Any]]) -> dict[str, list[Any]]:
    """Group a list of pairs according to their first elements.

    The result is dictionary, mapping each unique first element to the set of associated
    second elements (i.e. that appear together within a pair).
    """

    def first(item: tuple[Any, Any]) -> Any:
        return item[0]

    return {
        # since groupby will return the result of the `key` function in association to
        # the iterable elemnts, let's slice the iterable to retain just the second
        # elements - since our `key` is actually the first one
        name: [value for _, value in grouped]
        # groupby only groups adjacent items, so let's sort them first
        for name, grouped in groupby(sorted(items, key=first), key=first)
    }


def deduplicate_configs(configs: list[tuple[str, StrDict]]) -> dict[str, StrDict]:
    """Deduplicate port configurations.

    Configurations could be repeated after initial generation, for two different
    reasons:

        - because they appear multiple times in the original platform's parameters, e.g.
          since different LO objects are associated to channels connected to the same
          port
        - or because they can be reached through different paths, as in the case in
          which the LO object is a single one, but referenced by all associated channels

    then we need to ensure compatibility.

    This function is checking that all configurations targeted to the same object have
    compatible values, and raises otherwise.
    Compatible here means equal (by comparison) among values which are set. For a value
    which may appear multiple times is allowed to be unset some of them, in which case
    is implicitly set by the other occurences (as opposed to be compatible with its
    reset value).
    """

    def dedup(cfgs: list[StrDict], path: str) -> StrDict:
        """Deduplicate multiple configurations for the same path."""
        # concatenate all the configs items in a single list
        # - this could contain repeated keys
        items = [(k, v) for cfg in cfgs for k, v in cfg.items()]
        # ... and group it together according to their key
        grouped = groupitems(items)

        port = {}
        # check for values compatibility
        for k, vals in grouped.items():
            val = vals[0]
            if any(v != val for v in vals[1:]):
                raise ValueError(
                    f"Multiple inconsistent occurences of '{k}' for '{path}'\n"
                    f"Values:\n  {vals}"
                )
            # append to port configurations
            port[k] = val
        return port

    # configurations are grouped by port path (e.g. `out2` or `out0_in0`) and then the
    # values associated to this port are deduplicated
    return {path: dedup(config, path) for path, config in groupitems(configs).items()}
