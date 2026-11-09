"""Tests for TWPA pump support in the Qblox driver (CW and swept modes)."""

import numpy as np
import pytest

from qibolab._core.components import (
    AcquisitionChannel,
    IqChannel,
    IqConfig,
    OscillatorConfig,
)
from qibolab._core.execution_parameters import AcquisitionType, ExecutionParameters
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.config.port import PortAddress, PortConfig
from qibolab._core.instruments.qblox.config.sequencer import SequencerConfig
from qibolab._core.instruments.qblox.q1asm.ast_ import (
    Loop,
    SetAwgOffs,
    SetFreq,
    Stop,
    UpdParam,
    Wait,
    WaitSync,
)
from qibolab._core.instruments.qblox.sequence import Q1Sequence, compile
from qibolab._core.instruments.qblox.sequence.asm import Registers
from qibolab._core.instruments.qblox.twpa import (
    max_output_attenuation,
    twpa_attenuation_and_offset,
)
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper

# ---------------------------------------------------------------------------
# `Q1Sequence` CW and Swept TWPA
# ---------------------------------------------------------------------------


def test_q1sequence_cw():
    seq = Q1Sequence.cw()
    assert seq.is_continuous_wave is True
    assert len(seq.waveforms) == 1
    assert len(seq.acquisitions) == 0
    assert len(seq.program.elements) == 1
    assert isinstance(seq.program.elements[0].instruction, Stop)


def test_q1sequence_from_twpa_frequency_swept():
    options = ExecutionParameters(nshots=100, relaxation_time=50_000)
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([100e6, 200e6, 300e6]),
        channels=["twpa_ch"],
    )
    seq = Q1Sequence.from_twpa(
        options=options,
        sweepers=[[sweeper]],
        sampling_rate=1.0,
        channel="twpa_ch",
        duration=1000.0,
    )

    assert seq.is_continuous_wave is False
    assert len(seq.waveforms) == 0
    assert len(seq.acquisitions) == 0

    instructions = [line.instruction for line in seq.program.elements]
    assert any(isinstance(ins, UpdParam) for ins in instructions)
    assert any(isinstance(ins, WaitSync) for ins in instructions)
    assert any(isinstance(ins, Wait) and ins.duration == 1000 for ins in instructions)
    # Check no relaxation wait (duration 50_000) is present
    assert not any(
        isinstance(ins, Wait) and ins.duration == 50_000 for ins in instructions
    )
    # Check loop logic and SetFreq are present
    assert any(isinstance(ins, SetFreq) for ins in instructions)
    assert any(isinstance(ins, Loop) for ins in instructions)


def test_q1sequence_from_twpa_frequency_out_of_range():
    options = ExecutionParameters(nshots=100, relaxation_time=50_000)
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([6.0e9, 6.1e9, 6.2e9]),
        channels=["twpa_ch"],
    )
    with pytest.raises(ValueError, match="IF frequency must be a float between"):
        Q1Sequence.from_twpa(
            options=options,
            sweepers=[[sweeper]],
            sampling_rate=1.0,
            channel="twpa_ch",
            duration=1000.0,
        )


def test_q1sequence_from_twpa_offset_swept():
    options = ExecutionParameters(nshots=50, relaxation_time=100_000)
    sweeper = Sweeper(
        parameter=Parameter.offset,
        values=np.array([0.1, 0.2, 0.3]),
        channels=["twpa_ch"],
    )
    seq = Q1Sequence.from_twpa(
        options=options,
        sweepers=[[sweeper]],
        sampling_rate=1.0,
        channel="twpa_ch",
        duration=500.0,
    )

    assert seq.is_continuous_wave is False
    instructions = [line.instruction for line in seq.program.elements]
    assert any(isinstance(ins, UpdParam) for ins in instructions)
    assert any(isinstance(ins, WaitSync) for ins in instructions)
    assert any(isinstance(ins, Wait) and ins.duration == 500 for ins in instructions)
    assert not any(
        isinstance(ins, Wait) and ins.duration == 100_000 for ins in instructions
    )
    assert any(isinstance(ins, Loop) for ins in instructions)
    # the swept offset drives the first path only, as the static offset, such that the
    # pump amplitude is the same in both modes
    offsets = [ins for ins in instructions if isinstance(ins, SetAwgOffs)]
    assert len(offsets) > 0
    assert all(ins.value_1 == Registers.zero.value for ins in offsets)
    assert all(ins.value_0 != Registers.zero.value for ins in offsets)


# ---------------------------------------------------------------------------
# `compile` with TWPA
# ---------------------------------------------------------------------------


def test_compile_twpa_cw():
    ps = PulseSequence()
    options = ExecutionParameters(nshots=10, relaxation_time=1000)
    seqs = compile(
        sequence=ps,
        sweepers=[],
        options=options,
        sampling_rate=1.0,
        merged_vzs=True,
        twpas={"twpa": 1.0},
    )
    assert "twpa" in seqs
    assert seqs["twpa"].is_continuous_wave is True


def test_compile_twpa_swept():
    ps = PulseSequence()
    options = ExecutionParameters(nshots=10, relaxation_time=1000)
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([100e6, 200e6]),
        channels=["twpa"],
    )
    seqs = compile(
        sequence=ps,
        sweepers=[[sweeper]],
        options=options,
        sampling_rate=1.0,
        merged_vzs=True,
        twpas={"twpa": 1.0},
    )
    assert "twpa" in seqs
    assert seqs["twpa"].is_continuous_wave is False


# ---------------------------------------------------------------------------
# `SequencerConfig.build` for TWPA
# ---------------------------------------------------------------------------


def test_sequencer_config_build_twpa_cw():
    c = Cluster(
        address="addr",
        name="my_cluster",
        channels={"twpa": IqChannel(path="8/o1", lo="twpa/lo")},
        twpas={"twpa"},
    )
    address = PortAddress.from_path("8/o1")
    configs = {
        "twpa": IqConfig(frequency=6.5e9),
        "twpa/lo": OscillatorConfig(frequency=6.5e9, power=-10.0),
    }

    seq = Q1Sequence.cw()
    cfg = SequencerConfig.build(
        address=address,
        channel_id="twpa",
        channels=c.channels,
        configs=configs,
        acquisition=AcquisitionType.INTEGRATION,
        rf=True,
        sequence=seq,
    )

    assert cfg.sync_en is False
    assert cfg.cont_mode_en_awg_path0 is True
    assert cfg.cont_mode_en_awg_path1 is True
    assert cfg.offset_awg_path0 == 1.0
    assert cfg.nco_freq == 0
    assert cfg.mixer_corr_gain_ratio == 1.0
    assert cfg.mixer_corr_phase_offset_degree == 0.0


def test_sequencer_config_build_twpa_with_mixer():
    from qibolab._core.components import MixerOffsetConfig

    c = Cluster(
        address="addr",
        name="my_cluster",
        channels={"twpa": IqChannel(path="8/o1", lo="twpa/lo", mixer="mixer2")},
        twpas={"twpa"},
    )
    address = PortAddress.from_path("8/o1")
    configs = {
        "twpa": IqConfig(frequency=6.5e9),
        "twpa/lo": OscillatorConfig(frequency=6.5e9, power=-10.0),
        "mixer2": MixerOffsetConfig(offset_i=0.01, offset_q=-0.02),
    }

    seq = Q1Sequence.cw()
    cfg = SequencerConfig.build(
        address=address,
        channel_id="twpa",
        channels=c.channels,
        configs=configs,
        acquisition=AcquisitionType.INTEGRATION,
        rf=True,
        sequence=seq,
    )

    assert cfg.nco_freq == 0
    assert cfg.mixer_corr_gain_ratio == 1.0
    assert cfg.mixer_corr_phase_offset_degree == 0.0

    port = PortConfig.build(
        channel=c.channels["twpa"],
        config=configs["twpa"],
        in_=False,
        out=True,
        lo=configs["twpa/lo"],
        mixer=configs["mixer2"],
        is_qcm_non_rf_type=False,
    )
    assert port.offset_path0 == 0.01
    assert port.offset_path1 == -0.02


def test_sequencer_config_build_twpa_swept():
    c = Cluster(
        address="addr",
        name="my_cluster",
        channels={"twpa": IqChannel(path="8/o1", lo="twpa/lo")},
        twpas={"twpa"},
    )
    address = PortAddress.from_path("8/o1")
    configs = {
        "twpa": IqConfig(frequency=6.5e9, scale_q=0.9, phase_q=0.1),
        "twpa/lo": OscillatorConfig(frequency=6.5e9, power=-10.0),
    }

    options = ExecutionParameters(nshots=10, relaxation_time=1000)
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=np.array([100e6, 200e6]),
        channels=["twpa"],
    )
    seq = Q1Sequence.from_twpa(options, [[sweeper]], 1.0, "twpa", 100.0)

    cfg = SequencerConfig.build(
        address=address,
        channel_id="twpa",
        channels=c.channels,
        configs=configs,
        acquisition=AcquisitionType.INTEGRATION,
        rf=True,
        sequence=seq,
    )

    assert cfg.sync_en is True
    assert cfg.cont_mode_en_awg_path0 is None
    assert cfg.offset_awg_path0 == 0.0
    assert cfg.nco_freq == 0
    assert cfg.mixer_corr_gain_ratio == 0.9
    assert cfg.mixer_corr_phase_offset_degree == pytest.approx(np.degrees(0.1))


# ---------------------------------------------------------------------------
# `Cluster` attributes and resolution helpers
# ---------------------------------------------------------------------------


def test_cluster_twpa_fields_defaults():
    c = Cluster(address="addr", name="my_cluster")
    assert c.twpas == set()


def test_cluster_twpa_custom():
    c = Cluster(
        address="addr",
        name="my_cluster",
        channels={"twpa1": IqChannel(path="8/o1", lo="twpa1/lo")},
        twpas={"twpa1"},
    )
    assert c.twpas == {"twpa1"}


@pytest.mark.parametrize(
    "channels", [{}, {"twpa": AcquisitionChannel(path="8/i1")}], ids=["missing", "acq"]
)
def test_cluster_twpa_invalid(channels):
    with pytest.raises(ValueError, match="TWPAs have to be IQ channels"):
        Cluster(address="addr", name="my_cluster", channels=channels, twpas={"twpa"})


def test_cluster_twpa_channels_by_module():
    c = Cluster(
        address="addr",
        name="my_cluster",
        channels={
            "drive": IqChannel(path="8/o2"),
            "twpa": IqChannel(path="8/o1", lo="twpa/lo"),
            "twpa2": IqChannel(path="7/o1", lo="twpa2/lo", mixer="mixer2"),
        },
        twpas={"twpa", "twpa2"},
    )

    by_mod = c._channels_by_module
    assert 8 in by_mod
    assert 7 in by_mod
    assert [ch for ch, _ in by_mod[8]] == ["drive", "twpa"]
    assert [ch for ch, _ in by_mod[7]] == ["twpa2"]
    assert c._los["twpa"] == "twpa/lo"
    assert c._los["twpa2"] == "twpa2/lo"
    assert "twpa" not in c._mixers
    assert c._mixers["twpa2"] == "mixer2"


def test_compile_twpa_rejects_pulses():
    """Pulses on TWPA channels would be dropped, thus they are rejected."""
    from qibolab._core.pulses import Pulse, Rectangular

    ps = PulseSequence(
        [("twpa", Pulse(duration=40, amplitude=0.5, envelope=Rectangular()))]
    )
    options = ExecutionParameters(nshots=10, relaxation_time=1000)
    with pytest.raises(ValueError, match="TWPA channels.*twpa"):
        compile(
            sequence=ps,
            sweepers=[],
            options=options,
            sampling_rate=1.0,
            merged_vzs=True,
            twpas={"twpa"},
        )


# ---------------------------------------------------------------------------
# TWPA Attenuation and Offset Fine-tuning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_attenuation", [30, 60])
def test_twpa_attenuation_and_offset(max_attenuation):
    # Power = 0 dBm (0 dB att) -> (0, 1.0)
    assert twpa_attenuation_and_offset(0.0, max_attenuation) == (0, 1.0)

    # Even power (e.g. -10 dBm -> 10 dB att) -> (10, 1.0)
    assert twpa_attenuation_and_offset(-10.0, max_attenuation) == (10, 1.0)
    assert twpa_attenuation_and_offset(-4.0, max_attenuation) == (4, 1.0)

    # Odd / arbitrary power (e.g. -5 dBm -> 5 dB target att):
    # Port attenuation is 4 dB, offset fine-tunes 1 dB
    att, offset = twpa_attenuation_and_offset(-5.0, max_attenuation)
    assert att == 4
    assert np.isclose(offset, 10.0 ** (-1.0 / 20.0))

    # Fractional power (e.g. -11.5 dBm -> 11.5 dB target att):
    # Port attenuation is 10 dB, offset fine-tunes 1.5 dB
    att, offset = twpa_attenuation_and_offset(-11.5, max_attenuation)
    assert att == 10
    assert np.isclose(offset, 10.0 ** (-1.5 / 20.0))


@pytest.mark.parametrize("max_attenuation", [30, 60])
def test_twpa_attenuation_and_offset_beyond_maximum(max_attenuation):
    """Beyond the maximum attenuation of the port, the offset applies the remainder."""
    att, offset = twpa_attenuation_and_offset(-70.0, max_attenuation)
    assert att == max_attenuation
    assert np.isclose(offset, 10.0 ** (-(70.0 - max_attenuation) / 20.0))


def test_twpa_attenuation_and_offset_above_full_scale():
    with pytest.warns(UserWarning, match="exceeds the full scale"):
        assert twpa_attenuation_and_offset(5.0, 30) == (0, 1.0)


@pytest.mark.parametrize("power, att", [(-10.0, 10), (-5.0, 5), (-34.0, 34)])
def test_port_config_attenuation(power, att):
    """Ports apply the whole attenuation of their LO."""
    port = PortConfig.build(
        channel=IqChannel(path="8/o1"),
        config=IqConfig(frequency=6.5e9),
        in_=False,
        out=True,
        lo=OscillatorConfig(frequency=6.5e9, power=power),
        mixer=None,
        is_qcm_non_rf_type=False,
    )
    assert port.att == att


class _AttenuatorModule:
    """Module reporting the maximum attenuation of its outputs, as the Qblox driver."""

    def __init__(self, max_attenuation: int):
        self.max_attenuation = max_attenuation

    def _get_max_out_att_0(self) -> int:
        return self.max_attenuation

    def _get_max_out_att_1(self) -> int:
        return self.max_attenuation


class _Parameter:
    def __init__(self, values: dict, name: str):
        self.values, self.name = values, name

    def set(self, value):
        self.values[self.name] = value


class _ConfigurableModule(_AttenuatorModule):
    """Module recording the parameters set by the module configuration."""

    is_qcm_type = True
    is_qrm_type = False
    is_rf_type = True

    def __init__(self, max_attenuation: int):
        super().__init__(max_attenuation)
        self.values = {}

    @property
    def parameters(self):
        return _Parameters(self.values)

    def disconnect_outputs(self):
        pass


class _Parameters:
    def __init__(self, values: dict):
        self.values = values

    def __getitem__(self, name: str) -> _Parameter:
        return _Parameter(self.values, name)


def test_cluster_configure_modules_twpa_attenuation():
    """Only the ports of TWPA pumps apply part of their power, the rest being left to
    the offset."""
    c = Cluster(
        address="addr",
        name="my_cluster",
        channels={
            "drive": IqChannel(path="8/o1", lo="drive/lo"),
            "twpa": IqChannel(path="8/o2", lo="twpa/lo"),
        },
        twpas={"twpa"},
    )
    module = _ConfigurableModule(30)
    c.__dict__["_modules"] = {8: module}
    configs = {
        "drive": IqConfig(frequency=6e9),
        "drive/lo": OscillatorConfig(frequency=6e9, power=-35.0),
        "twpa": IqConfig(frequency=8e9),
        "twpa/lo": OscillatorConfig(frequency=8e9, power=-35.0),
    }
    modcfg = c._configure_modules(configs)[8]
    assert modcfg.ports["out0_att"] == 35
    assert modcfg.ports["out1_att"] == 30
    assert module.values["out1_att"] == 30


@pytest.mark.parametrize("max_attenuation", [30, 60])
def test_max_output_attenuation(max_attenuation):
    module = _AttenuatorModule(max_attenuation)
    assert max_output_attenuation(module, 1) == max_attenuation


def test_cluster_twpa_powers():
    """Each pump is split within the attenuation range of its own module."""
    c = Cluster(
        address="addr",
        name="my_cluster",
        channels={
            "twpa": IqChannel(path="8/o2", lo="twpa/lo"),
            "twpa2": IqChannel(path="10/o1", lo="twpa2/lo"),
            "twpa3": IqChannel(path="10/o2"),
        },
        twpas={"twpa", "twpa2", "twpa3"},
    )
    # QCM-RF II in slot 8, original QCM-RF in slot 10
    c.__dict__["_modules"] = {8: _AttenuatorModule(30), 10: _AttenuatorModule(60)}
    configs = {
        "twpa/lo": OscillatorConfig(frequency=8e9, power=-40.0),
        "twpa2/lo": OscillatorConfig(frequency=8e9, power=-40.0),
    }
    powers = c._twpa_powers(configs)
    # pumps without LO keep their full offset, and are thus omitted
    assert set(powers) == {"twpa", "twpa2"}
    assert powers["twpa"][0] == 30
    assert np.isclose(powers["twpa"][1], 10.0 ** (-10.0 / 20.0))
    assert powers["twpa2"] == (40, 1.0)
    # while all of them have an offset
    offsets = c._twpa_offsets(configs)
    assert offsets == {"twpa": powers["twpa"][1], "twpa2": 1.0, "twpa3": 1.0}


class _StoppableModule(_AttenuatorModule):
    def stop_sequencer(self):
        pass


class _ClosableCluster:
    def close(self):
        pass


def test_cluster_disconnect_clears_max_attenuations():
    """The maximum attenuations depend on the modules, possibly changed on
    reconnection."""
    c = Cluster(
        address="addr",
        name="my_cluster",
        channels={"twpa": IqChannel(path="8/o2", lo="twpa/lo")},
        twpas={"twpa"},
    )
    c.__dict__["_modules"] = {8: _StoppableModule(30)}
    c._cluster = _ClosableCluster()
    assert c._twpa_max_attenuations == {"twpa": 30}
    c.disconnect()
    assert "_twpa_max_attenuations" not in c.__dict__
    c.__dict__["_modules"] = {8: _StoppableModule(60)}
    assert c._twpa_max_attenuations == {"twpa": 60}


def test_mock_module_max_output_attenuation():
    from qibolab._core.instruments.qblox.mock.cluster import (
        MOCK_MAX_OUTPUT_ATTENUATION,
        MockModule,
    )

    module = MockModule(8)
    assert max_output_attenuation(module, 0) == MOCK_MAX_OUTPUT_ATTENUATION


def test_sequencer_config_build_twpa_arbitrary_power_cw():
    c = Cluster(
        address="addr",
        name="my_cluster",
        channels={"twpa": IqChannel(path="8/o1", lo="twpa/lo")},
        twpas={"twpa"},
    )
    address = PortAddress.from_path("8/o1")
    configs = {
        "twpa": IqConfig(frequency=6.5e9),
        "twpa/lo": OscillatorConfig(frequency=6.5e9, power=-5.0),
    }

    seq = Q1Sequence.cw()
    expected_offset = 10.0 ** (-1.0 / 20.0)
    cfg = SequencerConfig.build(
        address=address,
        channel_id="twpa",
        channels=c.channels,
        configs=configs,
        acquisition=AcquisitionType.INTEGRATION,
        rf=True,
        sequence=seq,
        cw_offset=expected_offset,
    )

    assert cfg.sync_en is False
    assert cfg.cont_mode_en_awg_path0 is True
    assert cfg.cont_mode_en_awg_path1 is True
    assert np.isclose(cfg.offset_awg_path0, expected_offset)


def test_q1sequence_from_twpa_offset_swept_scaled():
    options = ExecutionParameters(nshots=50, relaxation_time=100_000)
    sweeper = Sweeper(
        parameter=Parameter.offset,
        values=np.array([0.1, 0.2, 0.3]),
        channels=["twpa_ch"],
    )
    offset_scale = 10.0 ** (-1.0 / 20.0)
    seq = Q1Sequence.from_twpa(
        options=options,
        sweepers=[[sweeper]],
        sampling_rate=1.0,
        channel="twpa_ch",
        duration=500.0,
        offset=offset_scale,
    )

    from qibolab._core.instruments.qblox.q1asm.ast_ import Move
    from qibolab._core.instruments.qblox.sequence.asm import convert

    instructions = [line.instruction for line in seq.program.elements]
    expected_start = int(convert(0.1 * offset_scale, Parameter.offset))
    moves = [ins for ins in instructions if isinstance(ins, Move)]
    assert any(ins.source == expected_start for ins in moves)


def test_compile_twpa_offset_swept_scaled():
    ps = PulseSequence()
    options = ExecutionParameters(nshots=10, relaxation_time=1000)
    sweeper = Sweeper(
        parameter=Parameter.offset,
        values=np.array([0.1, 0.2]),
        channels=["twpa"],
    )
    offset_scale = 10.0 ** (-1.0 / 20.0)
    seqs = compile(
        sequence=ps,
        sweepers=[[sweeper]],
        options=options,
        sampling_rate=1.0,
        merged_vzs=True,
        twpas={"twpa": offset_scale},
    )
    assert "twpa" in seqs
    seq = seqs["twpa"]
    assert seq.is_continuous_wave is False

    from qibolab._core.instruments.qblox.q1asm.ast_ import Move
    from qibolab._core.instruments.qblox.sequence.asm import convert

    expected_start = int(convert(0.1 * offset_scale, Parameter.offset))
    instructions = [line.instruction for line in seq.program.elements]
    moves = [ins for ins in instructions if isinstance(ins, Move)]
    assert any(ins.source == expected_start for ins in moves)
