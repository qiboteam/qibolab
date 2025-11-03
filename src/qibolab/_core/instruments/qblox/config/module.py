from enum import Flag, auto
from typing import Annotated, Any, Literal, cast

import numpy as np
from qblox_instruments.qcodes_drivers.module import Module

from qibolab._core.components import Channel, OscillatorConfig
from qibolab._core.components.channels import AcquisitionChannel
from qibolab._core.components.configs import Configs, DcConfig, IqMixerConfig
from qibolab._core.components.filters import (
    ExponentialFilter,
    FiniteImpulseResponseFilter,
)
from qibolab._core.identifier import ChannelId
from qibolab._core.serialize import Model

from .port import PortAddress

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


def los(
    all: dict[ChannelId, str],
    configs: Configs,
    module_channels: list[ChannelId],
) -> dict[ChannelId, OscillatorConfig]:
    return {
        id_: cast(OscillatorConfig, configs[lo])
        for id_, lo in all.items()
        if id_ in {ch for ch in module_channels}
    }


class ModuleType(Flag):
    """Module types, used to declaratively restrict configurations scope."""

    QRM = auto()
    """QRM type module."""
    RF = auto()
    """Radio frequency module."""


class ModuleConfig(Model):
    ports: dict[str, Any]
    """Port-level configurations.

    These configurations do not exactly apply to the whole module, but not even to the
    individual sequencers.
    Instead, they are applying to the physical ports.

    So, they are defined at the module-level, but dynamically prefixed for the physical
    port.
    """
    # the following attributes are automatically processed and set
    scope_acq_trigger_mode_path0: Annotated[
        Literal["sequencer", "level"], ModuleType.QRM
    ] = "sequencer"
    """Scope trigger mode for path 0.

    .. note::

        Acquisition on external digital trigger currently unsupported.
    """
    scope_acq_trigger_mode_path1: Annotated[
        Literal["sequencer", "level"], ModuleType.QRM
    ] = "sequencer"
    """Scope trigger mode for path 0.

    Cf. :attr:`scope_acq_trigger_mode_path0`.
    """

    @classmethod
    def build(
        cls,
        channels: dict[ChannelId, Channel],
        configs: Configs,
        los: dict[ChannelId, OscillatorConfig],
        mixers: dict[ChannelId, IqMixerConfig],
        qrm: bool,
    ) -> "ModuleConfig":
        ports = {}

        # TODO: input mixers unused, but available in Qblox
        # at the moment, it would share Qibolab configurations with the output one,
        # but there is no reason why the attenuation should be the same
        # we would need a separate `AcquisitionChannel.mixer` entry

        def in_(iq: ChannelId) -> bool:
            return isinstance(channels[iq], AcquisitionChannel)

        # set lo frequencies
        for iq, lo in los.items():
            n = PortAddress.from_path(channels[iq].path).ports[0] - 1

            path = f"out{n}_in{n}" if qrm else f"out{n}"
            ports[f"{path}_lo_en"] = True
            ports[f"{path}_lo_freq"] = int(lo.frequency)
            ports[f"out{n}_att"] = int(lo.power)
            in__ = in_(iq)
            path_ = ("in" if in__ else "out") + str(n)
            if not in__:
                ports[f"{path_}_att"] = int(lo.power)

        # set mixer calibration
        for iq, mixer in mixers.items():
            n = PortAddress.from_path(channels[iq].path).ports[0] - 1
            in__ = in_(iq)
            path_ = ("in" if in__ else "out") + str(n)
            # cf. TODO above
            if not in__:
                ports[f"{path_}_offset_path0"] = mixer.offset_i
                ports[f"{path_}_offset_path1"] = mixer.offset_q

        for id, ch in channels.items():
            n = PortAddress.from_path(ch.path).ports[0] - 1
            config = configs[id]

            # offsets
            if isinstance(config, DcConfig):
                ports[f"out{n}_offset"] = config.offset * QCM_SWEEP_TO_OFFSET

            # first set all active channels to filter delay compensation by default
            # - for the FIR
            ports[f"out{n}_fir_config"] = "delay_comp"
            # - and for all exponentials
            for m in range(4):
                ports[f"out{n}_exp{m}_config"] = "delay_comp"

            # then let's enable them only for the available filters, and store the
            # coefficients
            if isinstance(config, DcConfig):
                filters = config.filters
                firs = [
                    f for f in filters if isinstance(f, FiniteImpulseResponseFilter)
                ]
                assert len(firs) <= 1, "At most 1 FIR filter available"
                if len(firs) == 1:
                    fir = firs[0]
                    ports[f"out{n}_fir_config"] = "enabled"
                    ports[f"out{n}_fir_coeffs"] = fir.coefficients

                exps = [f for f in filters if isinstance(f, ExponentialFilter)]
                assert len(exps) <= 4, "At most 4 exponential filters available"
                for m, exp in enumerate(exps):
                    ports[f"out{n}_exp{m}_config"] = "enabled"
                    ports[f"out{n}_exp{m}_amplitude"] = exp.amplitude
                    ports[f"out{n}_exp{m}_time_constant"] = exp.tau

        # set input attenuation
        if qrm:
            ports["in0_att"] = 0

        return cls(ports=ports)

    @staticmethod
    def _set_option(mod: Module, name: str, metadata: list, value: Any) -> None:
        # - avoid configuring not explicitly set values
        # - ports configurations have dynamical prefixes, they are handled separately
        if value is None or name == "ports":
            return

        flag = [m for m in metadata if isinstance(m, ModuleType)]
        if len(flag) > 0:
            assert len(flag) == 1
            modtype = flag[0]
            if ModuleType.QRM in modtype and not mod.is_qrm_type:
                return
            if ModuleType.RF in modtype and not mod.is_rf_type:
                return

        mod.set(name, value)

    def apply(self, mod: Module) -> None:
        """Configure module-wide settings."""
        # first disable all default sequencer connections
        mod.disconnect_outputs()

        if mod.is_qrm_type:
            # including input ones, if QRM
            mod.disconnect_inputs()

        for config, value in self.ports.items():
            mod.set(config, value)

        # apply all the other configurations
        for name, field in type(self).model_fields.items():
            self._set_option(mod, name, field.metadata, getattr(self, name))
