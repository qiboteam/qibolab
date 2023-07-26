.. _generalpurpose:

Platforms
=========

Qibolab provides support to different quantum laboratories.

Each lab is implemented using a :class:`qibolab.platform.Platform` object which implements basic features and connects instruments, qubits and channels.
Therefore, the ``Platform`` enables the user to interface with all
the required lab instruments at the same time with minimum effort.


Platform
""""""""

.. autoclass:: qibolab.platform.Platform
    :members:
    :member-order: bysource

MultiqubitPlatform (Qblox)
""""""""""""""""""""""""""

.. autoclass:: qibolab.platforms.multiqubit.MultiqubitPlatform
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

The :class:`qibolab.pulses.PulseSequence` class enables to combine different pulses
into a sequence through the ``add`` method.

Basic Pulse
"""""""""""

.. autoclass:: qibolab.pulses.Pulse
   :members:
   :member-order: bysource

.. autoclass:: qibolab.pulses.Waveform
   :members:
   :member-order: bysource

.. autoclass:: qibolab.pulses.PulseType
   :members:
   :member-order: bysource

Readout Pulse
"""""""""""""

.. autoclass:: qibolab.pulses.ReadoutPulse
   :members:
   :show-inheritance:
   :member-order: bysource

Drive Pulse
"""""""""""

.. autoclass:: qibolab.pulses.DrivePulse
   :members:
   :show-inheritance:
   :member-order: bysource

Flux Pulse
""""""""""

.. autoclass:: qibolab.pulses.FluxPulse
   :members:
   :show-inheritance:
   :member-order: bysource

Pulse Sequence
""""""""""""""

.. autoclass:: qibolab.pulses.PulseSequence
   :members:
   :member-order: bysource

.. _pulseshape:

Pulse shape
"""""""""""

Rectangular
-----------

.. autoclass:: qibolab.pulses.Rectangular
   :members:
   :member-order: bysource

Gaussian
--------

.. autoclass:: qibolab.pulses.Gaussian
   :members:
   :member-order: bysource

Drag
----

.. autoclass:: qibolab.pulses.Drag
   :members:
   :member-order: bysource

IIR
---

.. autoclass:: qibolab.pulses.IIR
   :members:
   :member-order: bysource

SNZ
---

.. autoclass:: qibolab.pulses.SNZ
   :members:
   :member-order: bysource

eCap
----

.. autoclass:: qibolab.pulses.eCap
   :members:
   :member-order: bysource

Data structures and sweeping
============================

.. autoclass:: qibolab.qubits.Qubit
   :members:
   :member-order: bysource

.. autoclass:: qibolab.qubits.QubitPair
   :members:
   :member-order: bysource

.. automodule:: qibolab.native
   :members:
   :member-order: bysource

.. autoclass:: qibolab.sweeper.Sweeper
   :members:
   :member-order: bysource

Symbolic Expression
===================

.. automodule:: qibolab.symbolic
   :members:
   :member-order: bysource

Results
=======

.. autoclass:: qibolab.result.IntegratedResults
   :members:
   :show-inheritance:
   :member-order: bysource


.. autoclass:: qibolab.result.AveragedIntegratedResults
   :members:
   :show-inheritance:
   :member-order: bysource

.. autoclass:: qibolab.result.RawWaveformResults
   :members:
   :show-inheritance:
   :member-order: bysource


.. autoclass:: qibolab.result.AveragedRawWaveformResults
   :members:
   :show-inheritance:
   :member-order: bysource

.. autoclass:: qibolab.result.SampleResults
   :members:
   :show-inheritance:
   :member-order: bysource


.. autoclass:: qibolab.result.AveragedSampleResults
   :members:
   :show-inheritance:
   :member-order: bysource


Abstract Instruments
====================

All instrument implementations should inherit the following class
and implement its abstract methods

.. autoclass:: qibolab.instruments.abstract.Instrument
   :members:
   :member-order: bysource

Instruments that contain arbitrary waveform generators (AWGs) and
can play pulses should inherit

.. autoclass:: qibolab.instruments.abstract.Controller
   :members:
   :member-order: bysource

Such instruments are connected to :class:`qibolab.instruments.port.Port`
objects, which provide the user interface for setting instrument settings.

.. autoclass:: qibolab.instruments.port.Port
   :members:
   :member-order: bysource

Auxiliary instruments that do not play instruments can inherit
:class:`qibolab.instruments.abstract.Instrument` directly.
An example are local oscillators

.. autoclass:: qibolab.instruments.oscillator.LocalOscillator
   :members:
   :member-order: bysource


Supperted Instruments
=====================

Qibolab provides drivers for different instruments including
local oscillators, qblox and FPGAs.

Qblox
"""""

.. automodule:: qibolab.instruments.qblox
   :members:
   :member-order: bysource

Quantum Machines
""""""""""""""""

.. automodule:: qibolab.instruments.qm
   :members:
   :member-order: bysource

Zurich Instruments
""""""""""""""""""

.. automodule:: qibolab.instruments.zhinst
   :members:
   :member-order: bysource

QuicSyn
"""""""

.. autoclass:: qibolab.instruments.icarusq.QuicSyn
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

RohdeSchwarz SGS100A
""""""""""""""""""""

.. autoclass:: qibolab.instruments.rohde_schwarz.SGS100A
   :members:
   :undoc-members:
   :member-order: bysource

Erasynth
""""""""

.. autoclass:: qibolab.instruments.erasynth.ERA
   :members:
   :undoc-members:
   :member-order: bysource

FPGA
""""

IcarusQFPGA
-----------

.. autoclass:: qibolab.instruments.icarusqfpga.IcarusQFPGA
   :members:
   :member-order: bysource

RFSoC
-----

.. autoclass:: qibolab.instruments.rfsoc.RFSoC
   :members:
   :member-order: bysource

Transpiler
==========

.. automodule:: qibolab.transpilers.abstract
   :members:
   :member-order: bysource

.. automodule:: qibolab.transpilers.fusion
   :members:
   :member-order: bysource

.. automodule:: qibolab.transpilers.gate_decompositions
   :members:
   :member-order: bysource

.. automodule:: qibolab.transpilers.unitary_decompositions
   :members:
   :member-order: bysource

.. automodule:: qibolab.transpilers.pipeline
   :members:
   :member-order: bysource

.. automodule:: qibolab.transpilers.routing
   :members:
   :member-order: bysource

.. automodule:: qibolab.transpilers.placer
   :members:
   :member-order: bysource

.. automodule:: qibolab.transpilers.star_connectivity
   :members:
   :member-order: bysource
