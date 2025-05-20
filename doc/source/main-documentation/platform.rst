.. _main_doc_platform:

Platforms
=========

Qibolab provides support to different quantum laboratories.

Each lab configuration is implemented using a :class:`qibolab.Platform` object which orchestrates instruments,
qubits and channels and provides the basic features for executing pulses.
Therefore, the ``Platform`` enables the user to interface with all
the required lab instruments at the same time with minimum effort.

The API reference section provides a description of all the attributes and methods of the ``Platform``. Here, let's focus on the main elements.

In the platform, the main methods can be divided in different sections:

- functions to coordinate the instruments (``connect``, ``disconnect``)
- a unique interface to execute experiments (``execute``)
- functions save parameters (``dump``)

The idea of the ``Platform`` is to serve as the only object exposed to the user, so that we can deploy experiments,
without any need of going into the low-level instrument-specific code.

For example, let's first define a platform (that we consider to be a single qubit platform) using the ``create`` method presented in :doc:`/tutorials/lab`:

.. testcode::  python

    from qibolab import create_platform

    platform = create_platform("dummy")

Now we connect to the instruments (note that we, the user, do not need to know which instruments are connected).

.. testcode::  python

    platform.connect()

We can easily access the names of channels and other components, and based on the name retrieve the corresponding configuration. As an example let's print some things:

.. note::
   If requested component does not exist in a particular platform, its name will be `None`, so watch out for such names, and make sure what you need exists before requesting its configuration.

.. testcode::  python

    drive_channel_id = platform.qubits[0].drive
    drive_channel = platform.channels[drive_channel_id]
    print(f"Drive channel name: {drive_channel_id}")
    print(f"Drive frequency: {platform.config(drive_channel_id).frequency}")

    drive_lo = drive_channel.lo
    if drive_lo is None:
        print(f"Drive channel {drive_channel_id} does not use an LO.")
    else:
        print(f"Name of LO for channel {drive_channel_id} is {drive_lo}")
        print(f"LO frequency: {platform.config(drive_lo).frequency}")

.. testoutput:: python
    :hide:

    Drive channel name: 0/drive
    Drive frequency: 4000000000.0
    Drive channel 0/drive does not use an LO.

Now we can create a simple sequence without explicitly giving any qubit specific parameter,
as these are loaded automatically from the platform, as defined in the corresponding ``parameters.json``:

.. testcode::  python

   from qibolab import Delay, PulseSequence
   import numpy as np

   ps = PulseSequence()
   qubit = platform.qubits[0]
   natives = platform.natives.single_qubit[0]
   ps.concatenate(natives.RX())
   ps.concatenate(natives.R(phi=np.pi / 2))
   ps.append((qubit.probe, Delay(duration=200)))
   ps.concatenate(natives.MZ())

Now we can execute the sequence on hardware:

.. testcode::  python

    from qibolab import (
        AcquisitionType,
        AveragingMode,
    )

    options = dict(
        nshots=1000,
        relaxation_time=10,
        fast_reset=False,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    results = platform.execute([ps], **options)

Finally, we can stop instruments and close connections.

.. testcode::  python

    platform.disconnect()

.. _main_doc_parameters:

Parameters
^^^^^^^^^^


.. _main_doc_dummy:

Dummy platform
^^^^^^^^^^^^^^

In addition to the real instruments presented in the :ref:`main_doc_instruments` section, Qibolab provides the :class:`qibolab.instruments.DummyInstrument`.
This instrument represents a controller that returns random numbers of the proper shape when executing any pulse sequence.
This instrument is also part of the dummy platform which is defined in :py:mod:`qibolab._core.dummy` and can be initialized as

.. testcode::  python

    from qibolab import create_platform

    platform = create_platform("dummy")

This platform is equivalent to real platforms in terms of attributes and functions, but returns just random numbers.
It is useful for testing parts of the code that do not necessarily require access to an actual quantum hardware platform.


.. _main_doc_channels:

Channels
--------

Channels play a pivotal role in connecting the quantum system to the control infrastructure.
Various types of channels are typically present in a quantum laboratory setup, including:

- the probe line (from device to qubit)
- the acquire line (from qubit to device)
- the drive line
- the flux line
- the TWPA pump line

Qibolab provides a general :class:`qibolab.Channel` object, as well as specializations depending on the channel role.
A channel is typically associated with a specific port on a control instrument, with port-specific properties like "attenuation" and "gain" that can be managed using provided getter and setter methods.
Channels are uniquely identified within the platform through their id.

The idea of channels is to streamline the pulse execution process.
The :class:`qibolab.PulseSequence` is a list of ``(channel_id, pulse)`` tuples, so that the platform identifies the channel that every pulse plays
and directs it to the appropriate port on the control instrument.

In setups involving frequency-specific pulses, a local oscillator (LO) might be required for up-conversion.
Although logically distinct from the qubit, the LO's frequency must align with the pulse requirements.
Qibolab accommodates this by enabling the assignment of a :class:`qibolab._core.instruments.oscillator.LocalOscillator` object
to the relevant channel :class:`qibolab.IqChannel`.
The controller's driver ensures the correct pulse frequency is set based on the LO's configuration.

Each channel has a :class:`qibolab._core.components.configs.Config` associated to it, which is a container of parameters related to the channel.
Configs also have different specializations that correspond to different channel types.
The platform holds default config parameters for all its channels, however the user is able to alter them by passing a config updates dictionary
when calling :meth:`qibolab.Platform.execute`.
The final configs are then sent to the controller instrument, which matches them to channels via their ids and ensures they are uploaded to the proper electronics.


.. _main_doc_qubits:

Qubits
------

The :class:`qibolab.Qubit` class serves as a container for the channels that are used to control the corresponding physical qubit.
These channels encompass distinct types, each serving a specific purpose:

- probe (measurement probe from controller device to the qubits)
- acquisition (measurement acquisition from qubits to controller)
- drive
- flux
- drive_extra (additional drive channels at different frequencies used to probe higher-level transition)

Some channel types are optional because not all hardware platforms require them.
For example, flux channels are typically relevant only for flux tunable qubits.

The :class:`qibolab.Qubit` class can also be used to represent coupler qubits, when these are available.

.. _main_doc_native:

Native
------

Each quantum platform supports a specific set of native gates, which are the quantum operations that have been calibrated.
If this set is universal any circuit can be transpiled and compiled to a pulse sequence which can then be deployed in the given platform.

:py:mod:`qibolab._core.native` provides data containers for holding the pulse parameters required for implementing every native gate.
The :class:`qibolab.Platform` provides a natives property that returns the :class:`qibolab._core.native.SingleQubitNatives`
which holds the single qubit native gates for every qubit and :class:`qibolab._core.native.TwoQubitNatives` for the two-qubit native gates of every qubit pair.
Each native gate is represented by a :class:`qibolab.PulseSequence` which contains all the calibrated parameters.

Typical single-qubit native gates are the Pauli-X gate, implemented via a pi-pulse which is calibrated using Rabi oscillations and the measurement gate,
implemented via a pulse sent in the readout line followed by an acquisition.
For a universal set of single-qubit gates, the RX90 (pi/2-pulse) gate is required,
which is implemented by halving the amplitude of the calibrated pi-pulse.

Typical two-qubit native gates are the CZ and iSWAP, with their availability being platform dependent.
These are implemented with a sequence of flux pulses, potentially to multiple qubits, and virtual Z-phases.
Depending on the platform and the quantum chip architecture, two-qubit gates may require pulses acting on qubits that are not targeted by the gate.
