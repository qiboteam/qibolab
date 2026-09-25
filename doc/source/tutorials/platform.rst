.. _tutorial_platform:

Building a Custom Platform
==========================

This tutorial guides you through creating a custom Qibolab platform from scratch.

What is a Platform?
-------------------

A :class:`.Platform` connects your quantum hardware to Qibolab. It defines:

- Which instruments control your QPU
- How qubits are physically connected
- What configurations each instrument needs
- What native gates are available

Structure
---------

A platform consists of two parts:

1. **Hardware** (:class:`.Hardware`): Instrument and qubit topology
2. **Parameters** (:class:`.Parameters`): Configurations and native gates

Step 1: Define Channels
-----------------------

Channels are physical communication lines. Start by determining which channels
your instruments provide:

.. code-block:: python

    from qibolab._core.components import IqChannel, DcChannel, AcquisitionChannel

    # Define channels provided by your controller
    drive_ch = IqChannel(path=(0,), mixer="mixer0", lo="lo_drive")
    flux_ch = DcChannel(path=(1,))
    probe_ch = IqChannel(path=(2,), lo="lo_probe")
    acq_ch = AcquisitionChannel(path=(3,), probe="probe")

Step 2: Create Instruments
---------------------------

Define instruments that will be part of your platform:

.. code-block:: python

    from qibolab._core.instruments.dummy import DummyInstrument

    # For a real platform, you would use your actual controller class
    controller = DummyInstrument(
        name="controller",
        address="192.168.1.100",
        channels={
            "drive": drive_ch,
            "flux": flux_ch,
            "probe": probe_ch,
            "acq": acq_ch,
        },
    )

Step 3: Define Qubits
---------------------

Group channels into logical qubits:

.. code-block:: python

    from qibolab._core.qubits import Qubit

    qubit0 = Qubit(
        drive="drive",
        flux="flux",
        probe="probe",
        acquisition="acq",
    )

    qubit1 = Qubit(
        drive="drive1",
        flux="flux1",
        probe="probe1",
        acquisition="acq1",
    )

Step 4: Create Hardware
-----------------------

Combine instruments and qubits into hardware:

.. code-block:: python

    from qibolab._core.platform import Hardware

    hardware = Hardware(
        instruments={"controller": controller},
        qubits={0: qubit0, 1: qubit1},
        couplers={},  # Add couplers if your platform has them
    )

Step 5: Define Configurations
------------------------------

Create configuration classes for your hardware:

.. code-block:: python

    from qibolab._core.components.configs import IqConfig, DcConfig, AcquisitionConfig

    # Store configurations for each channel
    configurations = {
        "drive": IqConfig(frequency=4.5e9, amplitude=1.0),
        "flux": DcConfig(amplitude=0.0),
        "probe": IqConfig(frequency=5.0e9, amplitude=0.1),
        "acq": AcquisitionConfig(frequency=5.0e9),
        # Repeat for other qubits...
    }

Step 6: Define Native Gates
----------------------------

Define native single-qubit and two-qubit gates:

.. code-block:: python

    from qibolab import Pulse, PulseSequence, Rectangular

    # Define RX(π/2) pulse
    rx_pulse = Pulse(
        duration=40,
        amplitude=0.5,
        relative_phase=0,
        envelope=Rectangular(),
    )

    # Store native gates with qubit associations
    # (Details depend on your platform's gate set)

Step 7: Create Parameters
--------------------------

Combine configurations and native gates into parameters:

.. code-block:: python

    from qibolab._core.parameters import Parameters

    parameters = Parameters(
        configurations=configurations,
        # Add native gates (platform-specific)
        nshots=1000,
        relaxation_time=100,
    )

Step 8: Create Platform
-----------------------

Combine hardware and parameters into a complete platform:

.. code-block:: python

    from qibolab._core.platform import Platform

    platform = Platform(
        name="my_custom_platform",
        parameters=parameters,
        instruments={"controller": controller},
        qubits={0: qubit0, 1: qubit1},
        couplers={},
    )

Step 9: Test and Use
--------------------

Test your platform:

.. code-block:: python

    # Connect
    platform.connect()

    # Define experiment
    sequence = platform.natives.single_qubit[0].RX()

    # Execute
    results = platform.execute([sequence])

    # Disconnect
    platform.disconnect()

Step 10: Save to Disk (Optional)
--------------------------------

Save your platform for reuse:

.. code-block:: python

    # Save parameters
    platform.dump()

    # Later, reload
    platform = create_platform("my_custom_platform")

Complete Example
----------------

For a complete working example, see the dummy platform implementation in
``src/qibolab/_core/dummy/platform.py``.

Next Steps
----------

- Implement a custom controller driver (see :ref:`tutorial_driver`)
- Define custom channel and configuration types
- Optimize native gate definitions for your hardware
- Calibrate gate parameters for accurate quantum operations
