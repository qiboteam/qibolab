import numpy as np

from qibolab._core.instruments.qblox.sequence.waveforms import waveforms
from qibolab._core.pulses import Custom, Pulse


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
