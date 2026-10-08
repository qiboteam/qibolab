.. _tutorials_sweeps:

Sweeping experiment parameters
==============================

Many pulse experiments are useful only when repeated over a parameter:
an amplitude scan calibrates a rotation, a frequency scan locates a
response, and a delay scan measures how a state evolves. Rebuilding and
submitting a sequence from Python at every point adds communication
overhead. A :class:`qibolab.Sweeper` describes the parameter values and
their targets so that a supported control system can execute the scan
as one experiment.

This tutorial uses the dummy platform and small scans to make the data
axes visible. It needs no hardware. The returned numbers are random
test data, **not a physical simulation**: the examples verify scan
definitions and result shapes, not Rabi oscillations, resonance curves,
or relaxation.

An amplitude scan on an existing pulse
--------------------------------------

Start with a calibrated native excitation followed by a fresh native
measurement. Retain the drive pulse to target the sweep and the readout
to retrieve its result. The objects must be the ones actually present
in the executed sequence.

.. testcode:: sweeps

    import numpy as np

    from qibolab import (
        AcquisitionType,
        AveragingMode,
        ExecutionParameters,
        Parameter,
        Pulse,
        PulseSequence,
        Sweeper,
        create_platform,
    )

    platform = create_platform("dummy")
    qubit = platform.qubits[0]
    natives = platform.natives.single_qubit[0]
    excitation = natives.RX()
    measurement = natives.MZ()
    sequence = excitation | measurement
    drive_pulse = next(
        event
        for channel, event in excitation
        if channel == qubit.drive and isinstance(event, Pulse)
    )
    readout = measurement.acquisitions[0][1]
    assert any(event is drive_pulse for _, event in sequence)

    amplitudes = np.linspace(0.0, 0.4, 5)
    amplitude_sweep = Sweeper(
        parameter=Parameter.amplitude,
        values=amplitudes,
        pulses=[drive_pulse],
    )
    single_shot_options = dict(
        nshots=8,
        relaxation_time=100,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )

    platform.connect()
    try:
        results = platform.execute([sequence], [[amplitude_sweep]], **single_shot_options)
    finally:
        platform.disconnect()

    iq = results[readout.id]
    assert iq.shape == (8, 5, 2)
    assert set(results) == {readout.id}

The nested brackets are part of the execution interface: the outer list
contains sweep **groups**, and the inner list contains sweepers that
advance together. A single sweeper in a single group creates one scan
axis. Here ``iq[shot, amplitude_index, 0]`` selects I at one point and
``iq[shot, amplitude_index, 1]`` selects Q. Averaging in Python with
``iq.mean(axis=0)`` would leave shape ``(5, 2)``; requesting hardware
averaging also removes the shot axis.

Amplitude values are absolute digital amplitudes, not scale factors
relative to the native amplitude. They are dimensionless and must lie
within ``[-1, 1]``. The native waveform supplies the starting shape and
duration; sweeping its amplitude does not mean the resulting values
remain calibrated rotations. Calling ``natives.RX()`` a second time
to obtain the sweeper target would produce a different pulse identifier,
not the pulse in this sequence.

For ``amplitude``, ``duration``, or ``relative_phase``, use ``pulses=[...]``.
Durations are in ns and relative phases in radians.
``Parameter.phase`` specifically targets ``VirtualZ`` instructions, also
in radians. A channel's ``frequency`` or ``offset`` instead uses
``channels=[...]``; frequencies are absolute values in Hz, and offsets
follow the channel configuration's physical units.
Provide exactly one target kind. A single sweeper can target several
pulses or channels, applying the same value to all of them at each point.

Make a frequency-amplitude grid
-------------------------------

To distinguish an amplitude response from detuning, scan the drive
frequency as well. Read the carrier from the channel configuration and
add offsets in Hz. A range is ``(start, stop, step)`` with the same
stop-exclusive semantics as ``numpy.arange``:

.. testcode:: sweeps

    center_frequency = platform.config(qubit.drive).frequency
    frequency_sweep = Sweeper(
        parameter=Parameter.frequency,
        range=(
            center_frequency - 2e6,
            center_frequency + 2e6,
            1e6,
        ),
        channels=[qubit.drive],
    )
    np.testing.assert_allclose(
        frequency_sweep.values - center_frequency,
        [-2e6, -1e6, 0, 1e6],
    )
    assert len(frequency_sweep) == 4

The scan spans offsets of -2, -1, 0, and +1 MHz; it does **not**
include +2 MHz. Use an explicit array, for example
``np.linspace(start, stop, count)``, when both endpoints must be
included. Supplying ``range`` stores the corresponding ``np.arange``
values and may allow a more efficient hardware implementation; when
both ``range`` and ``values`` are given, the range generates the values,
so prefer a single spelling.

Put the frequency and amplitude sweepers in separate groups to obtain
their Cartesian product:

.. testcode:: sweeps

    grid_sweepers = [[frequency_sweep], [amplitude_sweep]]

    platform.connect()
    try:
        grid_results = platform.execute([sequence], grid_sweepers, **single_shot_options)
    finally:
        platform.disconnect()

    grid_iq = grid_results[readout.id]
    print(grid_iq.shape)
    assert grid_iq[:, 2, :, :].shape == (8, 5, 2)
    assert grid_iq[:, :, 3, :].shape == (8, 4, 2)

.. testoutput:: sweeps

    (8, 4, 5, 2)

The first group is the outer scan loop. At each of the four frequencies,
the second group traverses all five amplitudes. Thus
``grid_iq[shot, frequency_index, amplitude_index, iq_component]`` is
the correct indexing order. The index 2 in the frequency axis corresponds
to the configured center frequency. Reversing the groups would instead
produce shape ``(8, 5, 4, 2)`` and exchange the meaning of those two axes.

This is a useful distinction from passing a list of independently built
sequences: the scan axes live inside each acquisition's array, whereas
different sequences still produce dictionary entries keyed by their
distinct readouts. There is no extra result axis for the number of
sequences.

Advance two parameters together
-------------------------------

Sometimes the desired path through parameter space is not a rectangular
grid. You might want to increase amplitude while changing phase, or scan
several probe frequencies with fixed relative detunings. Place those
sweepers in the **same** group to pair their values index by index.

.. testcode:: sweeps

    phases = np.linspace(0, np.pi, 5)
    phase_sweep = Sweeper(
        parameter=Parameter.relative_phase,
        values=phases,
        pulses=[drive_pulse],
    )
    paired_sweepers = [[amplitude_sweep, phase_sweep]]

    platform.connect()
    try:
        paired_results = platform.execute(
            [sequence],
            paired_sweepers,
            nshots=8,
            relaxation_time=100,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )
    finally:
        platform.disconnect()

    paired_iq = paired_results[readout.id]
    assert paired_iq.shape == (5, 2)
    assert list(zip(amplitudes, phases))[0] == (0.0, 0.0)
    assert np.allclose(list(zip(amplitudes, phases))[-1], (0.4, np.pi))

There are five paired settings, not 25 combinations. The amplitude and
phase sweepers share one axis; cyclic averaging removes the shot axis.
On hardware, cyclic averaging revisits the scan for each repetition,
whereas sequential averaging collects the repetitions at one setting
before moving to the next. Both averaged modes have the same shape.
Select a mode supported by the platform rather than assuming the modes
are interchangeable in their experimental noise behavior.

Parallel groups use zip-like length semantics in the core shape model:
their length is the **minimum** length of the contained sweepers.
Unequal arrays therefore do not imply broadcasting or a Cartesian
product. For example, a four-point frequency sweeper grouped with our
five-point amplitude sweeper describes a four-point axis:

.. testcode:: sweeps

    options_model = ExecutionParameters(**single_shot_options)
    assert options_model.results_shape([[frequency_sweep, amplitude_sweep]]) == (8, 4, 2)
    assert options_model.results_shape(grid_sweepers) == (8, 4, 5, 2)

Prefer equal-length arrays in a parallel group. Hardware can impose
stricter requirements, and relying on truncation can silently omit a
setting you intended to measure. Use nonempty groups and nonempty value
arrays; ``[]`` as the entire sweeper argument means no scan.

Follow the axes through acquisition modes
-----------------------------------------

Changing acquisition does not change the meaning or order of sweep
axes. The shot axis comes first only in ``SINGLESHOT`` mode, then one
axis per group, then the acquisition's own axes. Integration ends in
``(2,)`` for I/Q; discrimination has no trailing acquisition axis; raw
acquisition ends in ``(samples, 2)``.

The shape helper makes these distinctions concrete without executing
every mode:

.. testcode:: sweeps

    averaged_discrimination = ExecutionParameters(
        nshots=8,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    assert averaged_discrimination.results_shape(grid_sweepers) == (4, 5)

    single_discrimination = ExecutionParameters(
        nshots=8,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )
    assert single_discrimination.results_shape(grid_sweepers) == (8, 4, 5)

    single_raw = ExecutionParameters(
        nshots=8,
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=AveragingMode.SINGLESHOT,
    )
    assert single_raw.results_shape(grid_sweepers, samples=16) == (8, 4, 5, 16, 2)

Here ``samples=16`` is an explicitly supplied example sample count, not
a prediction for the native measurement used above. Raw sample counts
depend on the actual acquisition duration and sampling behavior.
For no sweeps, averaged discrimination is a scalar array with shape
``()``; with this grid it is a ``(4, 5)`` array. The
:ref:`results guide <main_doc_results>` covers the unswept cases too.

Varying time requires rechecking the schedule
---------------------------------------------

A duration sweep is often more useful on a ``Delay`` than on an output
pulse. For example, a relaxation experiment prepares a state and varies
the idle time before measurement:

.. testcode:: sweeps

    from qibolab import Delay

    wait = Delay(duration=40)
    waiting_stage = PulseSequence([(qubit.drive, wait)])
    delayed_measurement = natives.MZ()
    delayed_sequence = natives.RX() | waiting_stage | delayed_measurement
    delayed_readout = delayed_measurement.acquisitions[0][1]
    duration_sweep = Sweeper(
        parameter=Parameter.duration,
        values=np.array([40.0, 80.0, 120.0]),
        pulses=[wait],
    )

    platform.connect()
    try:
        delayed_results = platform.execute(
            [delayed_sequence],
            [[duration_sweep]],
            nshots=8,
            relaxation_time=100,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.SINGLESHOT,
        )
    finally:
        platform.disconnect()

    assert delayed_results[delayed_readout.id].shape == (8, 3)

The alignment-based ``|`` stages express the intended ordering even as
the wait changes. Check that the hardware's duration-sweep support
preserves that ordering; a schedule valid at one duration is not enough
to establish validity at every scan point. In particular, compiling
alignment markers into fixed delays before defining a duration sweep
can bake in the original timing.

``Parameter.duration_interpolated`` is another pulse-duration parameter,
intended for implementations that support interpolated-duration
sweeps; its availability and waveform treatment are hardware dependent.
Do not assume either duration-sweep mode is supported for every
instruction. Timing granularity, waveform memory, legal frequency
and offset ranges, and supported nesting or parallel combinations can
all constrain a real scan. A dummy execution checks none of those
physical restrictions.

Before a large hardware scan, validate the smallest supported version,
confirm the result axes, and choose realistic shot counts and reset
timing. The sweep description reduces submission overhead; it does not
replace calibration or the checks needed to make the experiment
physically meaningful.
