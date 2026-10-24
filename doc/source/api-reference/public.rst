.. _api_experiments:

Experiment API
==============

The objects below are available directly from ``qibolab``. Read the
:ref:`experiment guide <main_doc_experiment>` for the timing and result model,
or :doc:`../tutorials/pulses` for a worked example. This reference covers the
common API, not instrument-specific extensions.

Sequences and instructions
--------------------------

.. autoclass:: qibolab.PulseSequence
   :members:

.. autoclass:: qibolab.Pulse
   :members:

.. autoclass:: qibolab.Delay
   :members:

.. autoclass:: qibolab.VirtualZ
   :members:

.. autoclass:: qibolab.Align
   :members:

.. autoclass:: qibolab.Acquisition
   :members:

.. autoclass:: qibolab.Readout
   :members:

Instruction identity and copying
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All instruction types expose an ``id`` and a ``new()`` method. The inherited
members below are shown on ``Pulse`` as a representative instruction. ``new()``
copies the instruction rather than modifying it; ordinary shallow copies retain
the original identifier. ``Readout.id`` is the nested acquisition's identifier,
and ``Readout.new()`` also gives the nested acquisition and probe fresh identifiers.
See :ref:`main_doc_results` for using acquisition identities to retrieve data.

.. autoattribute:: qibolab.Pulse.id

.. automethod:: qibolab.Pulse.new

.. doctest:: instruction-identity

    >>> from qibolab import Pulse, Readout, Rectangular
    >>> pulse = Pulse(duration=16, amplitude=0.1, envelope=Rectangular())
    >>> pulse.model_copy().id == pulse.id
    True
    >>> fresh = pulse.new()
    >>> fresh.id != pulse.id and fresh.duration == pulse.duration
    True
    >>> readout = Readout.from_probe(pulse)
    >>> copied = readout.new()
    >>> copied.id == copied.acquisition.id
    True
    >>> copied.id != readout.id and copied.probe.id != readout.probe.id
    True

Envelopes
---------

Envelope methods produce baseband samples; the pulse supplies its amplitude
and duration. ``Pulse.i`` and ``Pulse.q`` take a sampling rate in GS/s, whereas
an envelope's methods take a number of samples.

.. autoclass:: qibolab.BaseEnvelope
   :members:

.. autoclass:: qibolab.Rectangular
   :members:

.. autoclass:: qibolab.Gaussian
   :members:

.. autoclass:: qibolab.GaussianSquare
   :members:

.. autoclass:: qibolab.Drag
   :members:

.. autoclass:: qibolab.Exponential
   :members:

.. autoclass:: qibolab.Snz
   :members:

.. autoclass:: qibolab.Custom
   :members:

Sweeps and execution
--------------------

.. autoclass:: qibolab.Parameter
   :members:

.. autoclass:: qibolab.Sweeper
   :members:

.. autoclass:: qibolab.ExecutionParameters
   :members:

.. autoclass:: qibolab.AcquisitionType
   :members:

.. autoclass:: qibolab.AveragingMode
   :members:

Identifiers and array types
---------------------------

Channel identifiers are strings. Qubit identifiers can be integers or strings,
and a qubit-pair identifier is a tuple of two qubit identifiers. Pulse identifiers
are UUIDs generated when instructions are created; acquisition identifiers key
the result dictionary returned by ``Platform.execute``.

``PulseLike`` and ``Envelope`` are discriminated unions of the corresponding
instruction and envelope models. ``ParallelSweepers`` is a list of sweepers
iterated together. ``Result``, ``Waveform``, and ``IqWaveform`` denote NumPy arrays;
their layout depends on the operation that produces them. In particular,
sampled pulse envelopes and acquisition results have different axis conventions.
