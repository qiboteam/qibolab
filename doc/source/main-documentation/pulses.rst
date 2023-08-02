Pulses
======

In Qibolab we have a dedicated API to pulses and pulses sequence, which
at the moment works for both qblox and FPGAs setups.

The main component of the API is the :class:`qibolab.pulses.Pulse` object,
which enables the user to code a pulse with specific parameters. We provide
also a special object for the ``ReadoutPulse`` given its importance when dealing
with a quantum hardware. Moreover, we supports different kinds of :ref:`pulseshape`.

The :class:`qibolab.pulses.PulseSequence` class enables to combine different pulses
into a sequence through the ``add`` method.
