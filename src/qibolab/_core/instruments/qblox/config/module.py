from enum import Flag, auto
from typing import Annotated, Any, Literal, cast

from qblox_instruments.qcodes_drivers.module import Module

from qibolab._core.components import Channel, OscillatorConfig
from qibolab._core.components.configs import Configs, IqMixerConfig
from qibolab._core.identifier import ChannelId
from qibolab._core.serialize import Model

__all__ = []


def los(
    all: dict[ChannelId, str],
    configs: Configs,
    module_channels: set[ChannelId],
) -> dict[ChannelId, OscillatorConfig]:
    return {
        id_: cast(OscillatorConfig, configs[lo])
        for id_, lo in all.items()
        if id_ in module_channels
    }


def mixers(
    all: dict[ChannelId, str],
    configs: Configs,
    module_channels: set[ChannelId],
) -> dict[ChannelId, IqMixerConfig]:
    # TODO: identical to the `.los()` function, deduplicate it please...
    return {
        id_: cast(IqMixerConfig, configs[mixer])
        for id_, mixer in all.items()
        if id_ in module_channels
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
        pass

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
