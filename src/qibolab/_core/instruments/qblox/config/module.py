from enum import Flag, auto
from typing import Annotated, Any, Literal, cast

from qblox_instruments.qcodes_drivers.module import Module

from qibolab._core.components import Channel, OscillatorConfig
from qibolab._core.components.channels import AcquisitionChannel, IqChannel
from qibolab._core.components.configs import Configs, IqMixerConfig
from qibolab._core.identifier import ChannelId
from qibolab._core.serialize import Model

from . import port

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
    twpa: dict[str, dict] | None = None
    """TWPA continuous-wave pump sources.

    Maps physical output port paths (e.g. ``"out1"``) to a dictionary with
    the pump tone configuration (``frequency``, ``lo_freq``, ``power``).
    Applied via :meth:`_apply_twpa_cw`, independent of the Q1ASM sequences.
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
        twpa_sources: dict[str, str] | None = None,
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
        # build TWPA continuous-wave pump sources, if any
        twpa: dict[str, dict] = {}
        if twpa_sources:
            for port_id, twpa_name in twpa_sources.items():
                if (osc_config := configs.get(twpa_name)) is not None:
                    twpa[f"out{port_id}"] = {
                        "frequency": int(osc_config.frequency),
                        "lo_freq": cls._twpa_lo_freq(channels, configs, port_id),
                        "power": osc_config.power,
                    }
                else:
                    raise ValueError(
                        f"TWPA source '{twpa_name}' (port {port_id}) not found in configs."
                    )

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
            },
            twpa=twpa or None,
        )

    @staticmethod
    def _twpa_lo_freq(
        channels: dict[ChannelId, Channel], configs: Configs, port_id: str
    ) -> int:
        """Resolve the LO frequency associated with a TWPA output port.

        Looks for a channel on the same physical port with a configured LO.
        Returns ``0`` if none is found.
        """
        port_num = int(port_id[1:])  # "o1" -> 1
        for ch in channels.values():
            ch_addr = port.PortAddress.from_path(ch.path)
            if ch_addr.ports[0] - 1 != port_num:
                continue
            probe_ch = (
                channels.get(ch.probe)
                if isinstance(ch, AcquisitionChannel) and ch.probe is not None
                else ch
            )
            if (
                isinstance(probe_ch, IqChannel)
                and probe_ch.lo is not None
                and (lo_cfg := configs.get(probe_ch.lo)) is not None
            ):
                return int(lo_cfg.frequency)
            break
        return 0

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

        mod.parameters[name].set(value)

    def apply(self, mod: Module) -> None:
        """Configure module-wide settings."""
        # first disable all default sequencer connections
        mod.disconnect_outputs()

        if mod.is_qrm_type:
            # including input ones, if QRM
            mod.disconnect_inputs()

        # apply all the configurations
        for name, field in type(self).model_fields.items():
            if name == "twpa":
                continue
            self._set_option(mod, name, field.metadata, getattr(self, name))

        # apply TWPA continuous-wave pumps, if any
        if self.twpa:
            for port_path, cfg in self.twpa.items():
                self._apply_twpa_cw(mod, port_path, cfg)

    @staticmethod
    def _apply_twpa_cw(mod: Module, port_path: str, cfg: dict) -> None:
        """Configure a dedicated sequencer in continuous-waveform mode for TWPA pump.

        Args:
            mod: the Qblox module.
            port_path: the physical output port path (e.g. ``"out1"``).
            cfg: dictionary with ``frequency``, ``lo_freq``, ``power`` keys.
        """
        from .sequencer import SequencerConfig

        port_id = int(port_path[3:])
        nco_freq = int(cfg["frequency"]) - int(cfg["lo_freq"])

        # Assign the last sequencer to the CW pump, to keep lower-indexed ones free
        # for Q1ASM sequences.
        seq_idx = len(mod.sequencers) - 1
        sequencer = mod.sequencers[seq_idx]

        # Configure the sequencer for CW mode via the `SequencerConfig`
        cw_cfg = SequencerConfig.build_cw(address=port_path, nco_freq=nco_freq)
        cw_cfg.apply(sequencer)

        # Enable the LO on the module for this port
        lo_param = f"out{port_id}_lo_en"
        if lo_param in mod.parameters:
            mod.parameters[lo_param].set(True)

        # Arm and start the CW pump
        mod.arm_sequencer(seq_idx)
        mod.start_sequencer()

    @staticmethod
    def disable_twpa_cw(mod: Module) -> None:
        """Disable continuous-waveform mode on all sequencers of a module."""
        for seq in mod.sequencers:
            for param in ("cont_mode_en_awg_path0", "cont_mode_en_awg_path1"):
                if param in seq.parameters:
                    seq.parameters[param].set(False)
