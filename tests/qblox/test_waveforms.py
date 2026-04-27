import numpy as np

from qibolab._core.instruments.qblox.sequence.waveforms import waveforms
from qibolab._core.pulses import Custom, Pulse, Rectangular
from qibolab._core.sweeper import Parameter, Sweeper


def test_waveforms_deduplicate_equal_components_across_distinct_iq_pairs():
    pulse_a = Pulse(
        duration=4,
        amplitude=1.0,
        envelope=Custom(
            i_=np.array([0.0, 0.2, 0.2, 0.0]),
            q_=np.array([0.0, 0.1, -0.1, 0.0]),
        ),
    )
    pulse_b = Pulse(
        duration=4,
        amplitude=1.0,
        envelope=Custom(
            i_=np.array([0.0, 0.2, 0.2, 0.0]),
            q_=np.array([0.0, -0.2, 0.2, 0.0]),
        ),
    )

    waveform_specs, indices_map = waveforms(
        sequence=[pulse_a, pulse_b],
        sampling_rate=1.0,
        amplitude_swept=set(),
        duration_swept={},
    )

    # Two unique Q components plus one shared I component.
    assert len(waveform_specs) == 3

    i_a_index, _ = indices_map[(pulse_a.id, 0)]
    i_b_index, _ = indices_map[(pulse_b.id, 0)]
    q_a_index, _ = indices_map[(pulse_a.id, 1)]
    q_b_index, _ = indices_map[(pulse_b.id, 1)]

    assert i_a_index == i_b_index
    assert q_a_index != q_b_index != i_a_index


def test_waveforms_deduplicate_across_distinct_lengths():
    pulse_a = Pulse(
        duration=6,
        amplitude=1.0,
        envelope=Custom(
            i_=np.array([0.0, 0.2, 0.2, 0.0, 0.1, 0.1]),
            q_=np.array([0.0, 0.1, -0.1, 0.0, 0.1, -0.1]),
        ),
    )

    pulse_b = Pulse(
        duration=4,
        amplitude=1.0,
        envelope=Custom(
            i_=np.array([0.0, 0.2, 0.2, 0.0]),
            q_=np.array([0.0, -0.2, 0.2, 0.0]),
        ),
    )

    waveform_specs, indices_map = waveforms(
        sequence=[pulse_a, pulse_b],
        sampling_rate=1.0,
        amplitude_swept=set(),
        duration_swept={},
    )

    # Two unique Q components plus two unique I components.
    assert len(waveform_specs) == 4

    i_a_index, _ = indices_map[(pulse_a.id, 0)]
    i_b_index, _ = indices_map[(pulse_b.id, 0)]
    q_a_index, _ = indices_map[(pulse_a.id, 1)]
    q_b_index, _ = indices_map[(pulse_b.id, 1)]

    assert q_a_index != q_b_index != i_a_index != i_b_index


def test_waveforms_duration_sweeper():

    # non-swept pulses with shared I component but distinct Q components
    pulse_a = Pulse(
        duration=4,
        amplitude=1.0,
        envelope=Custom(
            i_=np.array([0.0, 0.2, 0.2, 0.0]),
            q_=np.array([0.0, 0.2, 0.2, 0.0]),
        ),
    )

    # swept pulses
    pulse_c = Pulse(
        duration=4,
        amplitude=1.0,
        envelope=Rectangular(),
    )

    sweeper_c = Sweeper(
        parameter=Parameter.duration,
        range=(0, 2, 1),
        pulses=[pulse_c],
    )

    pulse_d = Pulse(
        duration=4,
        amplitude=1.0,
        envelope=Rectangular(),
    )

    sweeper_d = Sweeper(
        parameter=Parameter.duration,
        range=(0, 5, 2),
        pulses=[pulse_d],
    )

    waveform_specs, indices_map = waveforms(
        sequence=[pulse_a, pulse_c, pulse_d],
        sampling_rate=1.0,
        amplitude_swept=set(),
        duration_swept={
            pulse_c.id: sweeper_c,
            pulse_d.id: sweeper_d,
        },
    )

    # Two unique Q components plus one shared I component.
    assert len(waveform_specs) == (
        1
        + len(np.arange(*sweeper_c.irange)) * 2
        + len(np.arange(*sweeper_d.irange)) * 2
    )

    i_a_index, _ = indices_map[(pulse_a.id, 0)]
    q_a_index, _ = indices_map[(pulse_a.id, 1)]

    assert i_a_index == q_a_index

    swept_indices = list(indices_map.values())
    for ch in (0, 1):
        swept_indices.pop(swept_indices.index(indices_map[(pulse_a.id, ch)]))

    # ensure the swept_indices are their number is compatible with the sweeper ranges and all distinct
    assert (
        len(swept_indices)
        == len(np.arange(*sweeper_c.irange)) * 2 + len(np.arange(*sweeper_d.irange)) * 2
    )
    assert len(swept_indices) == len(set(swept_indices))
