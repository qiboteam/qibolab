.. _api_platforms:

Platform API
============

Most classes below are re-exported from ``qibolab``. Discovery and initialization
helpers are also available in ``qibolab.platform``. The
:ref:`platform guide <main_doc_platform>` explains how hardware descriptions and
serializable parameters work together.

Platforms and discovery
-----------------------

.. autoclass:: qibolab.Platform
   :members:

.. autoclass:: qibolab.Hardware
   :members:

.. autofunction:: qibolab.create_platform

.. autofunction:: qibolab.locate_platform

.. autofunction:: qibolab.platform.load_hardware

.. autofunction:: qibolab.platform.initialize_parameters

.. autofunction:: qibolab.platform.reset_parameters

.. autodata:: qibolab.platform.PLATFORM

.. autodata:: qibolab.platform.PLATFORMS_PATH

Qubits and channels
-------------------

.. autoclass:: qibolab.Qubit
   :members:

.. autoclass:: qibolab.Channel
   :members:

.. autoclass:: qibolab.DcChannel
   :members:

.. autoclass:: qibolab.IqChannel
   :members:

.. autoclass:: qibolab.AcquisitionChannel
   :members:

Parameters and configurations
-----------------------------

.. autoclass:: qibolab.Parameters
   :members:

.. autoclass:: qibolab.Config
   :members:

.. autoclass:: qibolab.ConfigKinds
   :members:

.. autoclass:: qibolab.DcConfig
   :members:

.. autoclass:: qibolab.IqConfig
   :members:

.. autoclass:: qibolab.MixerOffsetConfig
   :members:

.. autoclass:: qibolab.AcquisitionConfig
   :members:

.. autoclass:: qibolab.OscillatorConfig
   :members:

.. autoclass:: qibolab.LogConfig
   :members:

.. autoclass:: qibolab.ExponentialFilter
   :members:

.. autoclass:: qibolab.FiniteImpulseResponseFilter
   :members:

Native-operation containers
---------------------------

The types of ``platform.settings`` and ``platform.natives`` and the native
sequence containers live in core modules; they are not top-level imports.
Use the platform's accessors to retrieve these objects.

.. autoclass:: qibolab._core.parameters.Settings
   :members:

.. autoclass:: qibolab._core.parameters.NativeGates
   :members:

.. autoclass:: qibolab._core.native.Native
   :members:

.. autoclass:: qibolab._core.native.SingleQubitNatives
   :members:

.. autoclass:: qibolab._core.native.TwoQubitNatives
   :members:

Both native containers inherit ``ensure(name)``. It retrieves a defined native
template or raises ``MissingNative`` when that field is ``None``. Call the
returned template to make an executable sequence with fresh instruction
identifiers; retrieving it alone does not create a sequence.

.. automethod:: qibolab._core.native.SingleQubitNatives.ensure

.. doctest:: native-availability

    >>> from qibolab import create_platform
    >>> natives = create_platform("dummy").natives.single_qubit[0]
    >>> measurement = natives.ensure("MZ")
    >>> measurement is natives.MZ
    True
    >>> first, second = measurement(), measurement()
    >>> first.acquisitions[0][1].id != second.acquisitions[0][1].id
    True
