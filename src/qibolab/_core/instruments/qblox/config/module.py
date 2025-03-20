from enum import Flag, auto
from typing import Annotated, Any, Literal, cast

from qblox_instruments.qcodes_drivers.module import Module

from qibolab._core.components import Channel, OscillatorConfig
from qibolab._core.components.configs import Configs
from qibolab._core.identifier import ChannelId
from qibolab._core.serialize import Model

from .port import PortAddress

__all__ = []


def los(
    all: dict[ChannelId, str],
    configs: Configs,
    module_channels: list[tuple[ChannelId, PortAddress]],
) -> dict[ChannelId, OscillatorConfig]:
    return {
        id_: cast(OscillatorConfig, configs[lo])
        for id_, lo in all.items()
        if id_ in {ch[0] for ch in module_channels}
    }


class ModuleType(Flag):
    """Module types, used to declaratively restrict configurations scope."""

    QRM = auto()
    """QRM type module."""
    RF = auto()
    """Radio frequency module."""


class ModuleConfigs(Model):
    los: dict[str, Any]
    """Local oscillators configurations."""
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
    in0_att: Annotated[int, ModuleType.QRM] = 0
    """Input attenuation."""

    @classmethod
    def from_qibolab(
        cls,
        channels: dict[ChannelId, Channel],
        los: dict[ChannelId, OscillatorConfig],
        qrm: bool,
    ) -> "ModuleConfigs":
        los_ = {}
        # set lo frequencies
        for iq, lo in los.items():
            n = PortAddress.from_path(channels[iq].path).ports[0] - 1
            path = f"out{n}_in{n}" if qrm else f"out{n}"
            los_[f"{path}_lo_en"] = True
            los_[f"{path}_lo_freq"] = int(lo.frequency)
            los_[f"out{n}_att"] = int(lo.power)

        return cls(los=los_)

    @staticmethod
    def _set_option(mod: Module, name: str, metadata: list, value: Any) -> None:
        # - avoid configuring not explicitly set values
        # - los configurations have dynamical names, they are handled separately
        if value is None or name == "los":
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

        # only RF modules have LOs configured
        assert mod.is_rf_type or len(self.los) == 0
        for config, value in self.los.items():
            mod.set(config, value)

        # apply all the other configurations
        for name, field in self.model_fields.items():
            self._set_option(mod, name, field.metadata, getattr(self, name))
