from enum import Flag, auto
from typing import Annotated, Any, Literal, cast

from qblox_instruments.qcodes_drivers.module import Module

from qibolab._core.components import Channel, OscillatorConfig
from qibolab._core.components.channels import AcquisitionChannel
from qibolab._core.components.configs import Configs, IqMixerConfig
from qibolab._core.identifier import ChannelId
from qibolab._core.serialize import Model

from . import port
from .mixer import QbloxIqMixerConfig

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
    # TODO: support scope acquisition average

    @classmethod
    def build(
        cls,
        channels: dict[ChannelId, Channel],
        configs: Configs,
        los: dict[ChannelId, OscillatorConfig],
        mixers: dict[ChannelId, IqMixerConfig],
    ) -> "ModuleConfig":
        # generate port configurations as a dictionary
        def portconfig(*args, **kwargs) -> tuple[str, port.StrDict]:
            p = port.PortConfig.build(*args, **kwargs)
            return (p.path, p.model_dump(exclude_unset=True))

        # extend channel list to include probe channels
        # NOTE: the channel associated is still an `AcquisitionChannel`, to retain the
        # connection to the readout operation; this is later used to prevent separate
        # configuration of the "output LO", since there is only one LO for the probe and
        # acquisition (i.e. read-out and read-in)
        # since the identifier is the one of the probe channel, all retrieved
        # configurations will be related to the channel
        all_channels: list[tuple[ChannelId, Channel]] = list(channels.items()) + [
            (ch.probe, ch)
            for ch in channels.values()
            if isinstance(ch, AcquisitionChannel) and ch.probe is not None
        ]

        # since the configurations for the same path could be generated from multiple
        # channels, we keep a list of pairs, instead of a dictionary, to allow for
        # repeated keys, that will be merged later
        ports = [
            (path, port)
            for path, port in (
                portconfig(
                    channel=ch,
                    config=configs[id],
                    in_=in_,
                    out=out,
                    lo=los.get(id),
                    mixer=mixers.get(id),
                )
                # scrape all channels for port configurations
                for id, ch in all_channels
                # attempt all possible port usage - the `PortConfig` builder contains
                # all the logic to decide which is actually relevant for the given
                # channel
                for (in_, out) in [(True, False), (False, True), (True, True)]
            )
            # only retain non-empty configurations
            if len(port) > 0
        ]
        # since port configurations can be set or referenced through multiple paths,
        # let's check consistency, and deduplicate them
        ports = port.deduplicate_configs(ports)

        return cls(
            # since in Qblox port configurations are actually module configurations, we
            # "unroll" them here, just merging all the configurations for the present
            # module in a single dictionary, in which port configurations are just
            # prefixed by their path
            ports={
                f"{path}_{k}": v
                for path, configs in ports.items()
                for k, v in configs.items()
            }
        )

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
