.. _generalpurpose:

Platforms
=========

Qibolab provides support to different quantum laboratories.

Each lab is implemented using a custom ``Platform`` class
which inherits the :class:`qibolab.platforms.abstract.AbstractPlatform`
which implements basic features including how to connect to the platform,
how to start the instruments and how to run your model on the platform.
Therefore, the ``Platform`` enables the user to interface with all
the required lab instruments at the same time with minimum effort.


Abstract platform
"""""""""""""""""

.. autoclass:: qibolab.platforms.abstract.AbstractPlatform
    :members:
    :member-order: bysource

QBloxPlatform
"""""""""""""

.. autoclass:: qibolab.platforms.qbloxplatform.QBloxPlatform
   :members:
   :member-order: bysource

ICPlatform
""""""""""

.. autoclass:: qibolab.platforms.icplatform.ICPlatform
   :members:
   :member-order: bysource

_______________________

Pulses
======

In Qibolab we have a dedicated API to pulses and pulses sequence, which
at the moment works for both qblox and FPGAs setups.

The main component of the API is the :class:`qibolab.pulses.Pulse` object,
which enables the user to code a pulse with specific parameters. We provide
also a special object for the ``ReadoutPulse`` given its importance when dealing
with a quantum hardware. Moreover, we supports different kinds of :ref:`pulseshape`.

The :class:`qibolab.circuit.PulseSequence` class enables to combine different pulses
into a sequence through the ``add`` method.

Basic Pulse
"""""""""""

.. autoclass:: qibolab.pulses.Pulse
   :members:
   :member-order: bysource

Readout Pulse
"""""""""""""

.. autoclass:: qibolab.pulses.ReadoutPulse
   :members:
   :show-inheritance:
   :member-order: bysource

Pulse Sequence
""""""""""""""

.. autoclass:: qibolab.circuit.PulseSequence
   :members:
   :member-order: bysource

.. _pulseshape:

Pulse shape
"""""""""""

Rectangular
-----------

.. autoclass:: qibolab.pulse_shapes.Rectangular
   :members:
   :member-order: bysource

Gaussian
--------

.. autoclass:: qibolab.pulse_shapes.Gaussian
   :members:
   :member-order: bysource

Drag
----

.. autoclass:: qibolab.pulse_shapes.Drag
   :members:
   :member-order: bysource

SWIPHT
------

.. autoclass:: qibolab.pulse_shapes.SWIPHT
   :members:
   :member-order: bysource

Data structures and sweeping
============================

.. TODO: Add documentation

.. autoclass:: qibolab.platforms.abstract.Qubit
   :members:
   :member-order: bysource

.. autoclass:: qibolab.result.ExecutionResults
   :members:
   :member-order: bysource


.. autoclass:: qibolab.result.AveragedResults
   :members:
   :member-order: bysource

.. autoclass:: qibolab.sweeper.Sweeper
   :members:
   :member-order: bysource

Instruments supported
=====================

Qibolab currently supports different instruments including
local oscillators, qblox and FPGAs.

Qblox
"""""

GenericPulsar
-------------

.. autoclass:: qibolab.instruments.qblox.GenericPulsar
   :members:
   :member-order: bysource

PulsarQCM
---------

.. autoclass:: qibolab.instruments.qblox.PulsarQCM
   :members:
   :member-order: bysource

PulsarQRM
---------

.. autoclass:: qibolab.instruments.qblox.PulsarQRM
   :members:
   :member-order: bysource

QuicSyn
"""""""

.. autoclass:: qibolab.instruments.icarusq.QuicSyn
   :members:
   :undoc-members:
   :member-order: bysource

RohdeSchwarz SGS100A
""""""""""""""""""""

.. autoclass:: qibolab.instruments.rohde_schwarz.SGS100A
   :members:
   :undoc-members:
   :member-order: bysource

Tektronix AWG5204
"""""""""""""""""

.. autoclass:: qibolab.instruments.icarusq.TektronixAWG5204
   :members:
   :member-order: bysource

MiniCircuit RCDAT-8000-30
"""""""""""""""""""""""""

.. autoclass:: qibolab.instruments.icarusq.MCAttenuator
   :members:
   :member-order: bysource

FPGA
""""

ATS9371
-------

.. autoclass:: qibolab.instruments.ATS9371.AlazarTech_ATS9371
   :members:
   :member-order: bysource

.. autoclass:: qibolab.instruments.icarusq.AlazarADC
   :members:
   :member-order: bysource

RFSoC
---------

.. autoclass:: qibolab.instruments.rfsoc.RFSoC
   :members:
   :member-order: bysource
