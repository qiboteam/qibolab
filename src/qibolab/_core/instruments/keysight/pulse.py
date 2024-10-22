"""Utils for pulse handling."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Union

import keysight.qcs as qcs  # pylint: disable=E0401

from qibolab._core.pulses import Drag, Envelope, Gaussian, PulseId, Rectangular
from qibolab._core.pulses.pulse import PulseLike

NS_TO_S = 1e-9

__all__ = [
    "process_acquisition_channel_pulses",
    "process_iq_channel_pulses",
    "process_dc_channel_pulses",
]


def generate_qcs_envelope(shape: Envelope) -> qcs.Envelope:
    """Converts a Qibolab pulse envelope to a QCS Envelope object."""
    if isinstance(shape, Rectangular):
        return qcs.ConstantEnvelope()

    elif isinstance(shape, (Gaussian, Drag)):
        return qcs.GaussianEnvelope(shape.rel_sigma)

    else:
        # TODO: Rework this code to support other Qibolab pulse envelopes
        # raw_envelope = shape.i(num_samples) + 1j * shape.q(num_samples)
        # return qcs.ArbitraryEnvelope(
        #    times=np.linspace(0, 1, num_samples), amplitudes=raw_envelope
        # )
        raise Exception("Envelope not supported")


def process_acquisition_channel_pulses(
    program: qcs.Program,
    pulses: Iterable[PulseLike],
    frequency: Union[float, qcs.Scalar],
    virtual_channel: qcs.Channels,
    probe_virtual_channel: qcs.Channels,
    sweeper_pulse_map: defaultdict[PulseId, dict[str, qcs.Scalar]],
    classifier: qcs.Classifier = None,
):
    """Processes Qibolab pulses on the acquisition channel into QCS hardware
    instructions and adds it to the current program.

    Arguments:
        program (qcs.Program): Program object for the current sequence.
        pulses (Iterable[PulseLike]): Array of pulse objects to be processed.
        frequency (Union[float, qcs.Scalar]): Frequency of the channel.
        virtual_channel (qcs.Channels): QCS virtual digitizer channel.
        probe_virtual_channel (qcs.Channels): QCS virtual AWG channel connected to the digitzer.
        sweeper_pulse_map (defaultdict[PulseId, dict[str, qcs.Scalar]]): Map of pulse ID to map of parameter
        to be swept and corresponding QCS variable.
    """

    for pulse in pulses:
        sweep_param_map = sweeper_pulse_map.get(pulse.id, {})

        if pulse.kind == "delay":
            qcs_pulse = qcs.Delay(
                sweep_param_map.get("duration", pulse.duration * NS_TO_S)
            )
            program.add_waveform(qcs_pulse, virtual_channel)
            program.add_waveform(qcs_pulse, probe_virtual_channel)

        elif pulse.kind == "acquisition":
            duration = sweep_param_map.get("duration", pulse.duration * NS_TO_S)
            program.add_acquisition(duration, virtual_channel)

        elif pulse.kind == "readout":
            sweep_param_map = sweeper_pulse_map.get(pulse.probe.id, {})
            qcs_pulse = qcs.RFWaveform(
                duration=sweep_param_map.get(
                    "duration", pulse.probe.duration * NS_TO_S
                ),
                envelope=generate_qcs_envelope(pulse.probe.envelope),
                amplitude=sweep_param_map.get("amplitude", pulse.probe.amplitude),
                rf_frequency=frequency,
                instantaneous_phase=sweep_param_map.get(
                    "relative_phase", pulse.probe.relative_phase
                ),
            )
            integration_filter = qcs.IntegrationFilter(qcs_pulse)
            program.add_waveform(qcs_pulse, probe_virtual_channel)
            program.add_acquisition(integration_filter, virtual_channel, classifier)


def process_iq_channel_pulses(
    program: qcs.Program,
    pulses: Iterable[PulseLike],
    frequency: Union[float, qcs.Scalar],
    virtual_channel: qcs.Channels,
    sweeper_pulse_map: defaultdict[PulseId, dict[str, qcs.Scalar]],
):
    """Processes Qibolab pulses on the IQ channel into QCS hardware
    instructions and adds it to the current program.

    Arguments:
        program (qcs.Program): Program object for the current sequence.
        pulses (Iterable[PulseLike]): Array of pulse objects to be processed.
        frequency (Union[float, qcs.Scalar]): Frequency of the channel.
        virtual_channel (qcs.Channels): QCS virtual RF AWG channel.
        sweeper_pulse_map (defaultdict[PulseId, dict[str, qcs.Scalar]]): Map of pulse ID to map of parameter
        to be swept and corresponding QCS variable.
    """
    qcs_pulses = []
    for pulse in pulses:
        sweep_param_map = sweeper_pulse_map.get(pulse.id, {})

        if pulse.kind == "delay":
            qcs_pulse = qcs.Delay(
                sweep_param_map.get("duration", pulse.duration * NS_TO_S)
            )
        elif pulse.kind == "virtualz":
            qcs_pulse = qcs.PhaseIncrement(
                phase=sweep_param_map.get("relative_phase", pulse.phase)
            )
        elif pulse.kind == "pulse":
            qcs_pulse = qcs.RFWaveform(
                duration=sweep_param_map.get("duration", pulse.duration * NS_TO_S),
                envelope=generate_qcs_envelope(pulse.envelope),
                amplitude=sweep_param_map.get("amplitude", pulse.amplitude),
                rf_frequency=frequency,
                instantaneous_phase=sweep_param_map.get(
                    "relative_phase", pulse.relative_phase
                ),
            )
            if pulse.envelope.kind == "drag":
                qcs_pulse = qcs_pulse.drag(coeff=pulse.envelope.beta)
        else:
            raise ValueError("Unrecognized pulse type", pulse.kind)

        qcs_pulses.append(qcs_pulse)

    program.add_waveform(qcs_pulses, virtual_channel)


def process_dc_channel_pulses(
    program: qcs.Program,
    pulses: Iterable[PulseLike],
    virtual_channel: qcs.Channels,
    sweeper_pulse_map: defaultdict[PulseId, dict[str, qcs.Scalar]],
):
    """Processes Qibolab pulses on the DC channel into QCS hardware
    instructions and adds it to the current program.

    Arguments:
        program (qcs.Program): Program object for the current sequence.
        pulses (Iterable[PulseLike]): Array of pulse objects to be processed.
        virtual_channel (qcs.Channels): QCS virtual baseband AWG channel.
        sweeper_pulse_map (defaultdict[PulseId, dict[str, qcs.Scalar]]): Map of pulse ID to map of parameter
        to be swept and corresponding QCS variable.
    """
    qcs_pulses = []
    for pulse in pulses:
        sweep_param_map = sweeper_pulse_map.get(pulse.id, {})
        if pulse.kind == "delay":
            qcs_pulse = qcs.Delay(
                sweep_param_map.get("duration", pulse.duration * NS_TO_S)
            )
        elif pulse.kind == "pulse":
            qcs_pulse = qcs.DCWaveform(
                duration=sweep_param_map.get("duration", pulse.duration * NS_TO_S),
                envelope=generate_qcs_envelope(pulse.envelope),
                amplitude=sweep_param_map.get("amplitude", pulse.amplitude),
            )
        else:
            raise ValueError("Unrecognized pulse type", pulse.kind)
        qcs_pulses.append(qcs_pulse)

    program.add_waveform(qcs_pulses, virtual_channel)
