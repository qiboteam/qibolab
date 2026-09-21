.. _main_doc_driver_api:

Driver API
==========

The Driver API defines how to integrate new instruments with Qibolab.
Drivers handle the compilation of experiments and management of instrument communication.

Overview
--------

A driver is a Python class that:

1. Inherits from :class:`._core.instruments.abstract.Instrument` (or :class:`.Controller`)
2. Implements connection/disconnection to the physical instrument
3. For controllers, implements experiment execution (compilation and upload)

Qibolab ships with drivers for several platforms:

- **Controllers**: Quantum Machines (QM), Qblox, Qibosoq RFSoC, Dummy
- **Supporting**: ERASynth (LO), Rohde & Schwarz (LO), QRNG

Instrument Base Class
---------------------

All instruments inherit from :class:`._core.instruments.abstract.Instrument`:

.. code-block:: python

    from qibolab._core.instruments.abstract import Instrument, InstrumentSettings


    class MyInstrument(Instrument):
        """Custom instrument driver."""

        address: str  # Network address (e.g., "192.168.1.100")
        settings: InstrumentSettings | None = None

        def connect(self):
            """Establish connection to physical instrument."""
            # Implement connection logic
            pass

        def disconnect(self):
            """Close connection."""
            # Implement disconnection logic
            pass

        def setup(self, *args, **kwargs):
            """Configure instrument (optional, for non-controllers)."""
            pass

**Key Attributes**

- ``address``: Network address for communication
- ``settings``: Optional settings saved in platform runcard
- ``channels`` (controllers only): Mapping of channel IDs to :class:`.Channel` objects

**Key Methods**

- ``connect()``: Establish connection
- ``disconnect()``: Close connection
- ``setup()``: Configure non-controller instruments

Controller Class
----------------

Controllers execute experiments. They inherit from :class:`.Instrument` and add:

.. code-block:: python

    from qibolab._core.instruments.abstract import Controller
    from qibolab._core.components import Config, Channel


    class MyController(Controller):
        """Custom pulse controller."""

        channels: dict[ChannelId, Channel] = Field(default_factory=dict)

        @property
        def sampling_rate(self) -> float:
            """Sampling rate in GSps (giga samples per second)."""
            return 5.0  # Example: 5 GSps

        def play(
            self,
            configs: dict[str, Config],
            sequences: list[PulseSequence],
            options: ExecutionParameters,
            sweepers: list[ParallelSweepers],
        ) -> dict[PulseId, Result]:
            """Execute experiment and return results."""
            # Compile sequences to hardware instructions
            # Apply configurations
            # Upload to hardware
            # Run experiment
            # Download results
            # Return mapping of acquisition pulse IDs to results
            pass

**Key Methods**

- ``sampling_rate`` (property): Sampling rate in GSps
- ``play()``: Execute experiment with given sequences and options

Channel Definition
------------------

Channels define how instruments control the QPU. Qibolab provides base types:

- :class:`.DcChannel`: Direct current (no modulation)
- :class:`.IqChannel`: Modulated IQ (requires mixer, LO)
- :class:`.AcquisitionChannel`: Measurement readout

Define custom channels by subclassing:

.. code-block:: python

    from qibolab._core.components import IqChannel


    class CustomChannel(IqChannel):
        """Platform-specific channel."""

        custom_param: str = "default"

Configuration Classes
---------------------

Configurations hold runtime parameters. They must inherit from :class:`.Config`:

.. code-block:: python

    from qibolab._core.components.configs import Config


    class MyConfig(Config):
        """Platform-specific configuration."""

        frequency: float  # Frequency in Hz
        power: float = -20  # Power in dBm
        # Add any custom parameters

Instrument-specific configs extend base types:

.. code-block:: python

    from qibolab._core.components.configs import IqConfig


    class MyIqConfig(IqConfig):
        """Extended IQ config with custom parameters."""

        mixer_correction: float = 0.0

Compilation Workflow
--------------------

The ``play()`` method should:

1. **Receive inputs**:
   - ``configs``: Configuration dict (str -> Config)
   - ``sequences``: List of :class:`.PulseSequence` to execute
   - ``options``: :class:`.ExecutionParameters`
   - ``sweepers``: List of :class:`.ParallelSweepers` for real-time sweeps

2. **Compile sequences**:
   - Convert Qibolab pulses to hardware instructions
   - Apply configurations (frequency, amplitude, etc.)
   - Handle sweepers (native or unroll as sequences)
   - Optimize for hardware constraints

3. **Upload and execute**:
   - Upload compiled program to hardware
   - Run experiment
   - Collect results

4. **Return results**:
   - Dictionary mapping acquisition pulse IDs to numpy arrays
   - Shape should match :meth:`.ExecutionParameters.results_shape`

Example
-------

Here's a simplified example of a custom controller:

.. code-block:: python

    from qibolab._core.instruments.abstract import Controller
    from qibolab._core.components import IqChannel, IqConfig
    from pydantic import Field


    class SimplePulseController(Controller):
        """Minimal controller example."""

        channels: dict = Field(default_factory=dict)

        @property
        def sampling_rate(self) -> float:
            return 1.0  # 1 GSps

        def play(self, configs, sequences, options, sweepers):
            import numpy as np

            # Compile: convert sequences to hardware format
            # (implementation depends on your hardware)

            # Execute: run on hardware
            # (connect to hardware, upload, trigger, wait for results)

            # Return dummy results for demonstration
            results = {}
            for seq in sequences:
                for ch, pulse in seq.acquisitions:
                    pulse_id = pulse.id
                    shape = options.results_shape(sweepers)
                    results[pulse_id] = np.zeros(shape)

            return results

Best Practices
--------------

1. **Leverage Pydantic**: Use Pydantic models for validation
2. **Document parameters**: Clear docstrings for configuration classes
3. **Handle errors gracefully**: Meaningful error messages for configuration issues
4. **Optimize compilation**: Cache compiled programs when possible
5. **Test offline**: Provide a dummy/emulated mode for testing without hardware
6. **Support real-time sweepers**: Implement hardware sweepers when possible
7. **Validate channels**: Ensure all referenced channels exist

Extending Existing Drivers
---------------------------

To extend an existing driver (e.g., add custom channels or configs):

1. Subclass the channel or config class
2. Add instrument-specific parameters
3. Update the platform to use your custom classes
4. Document changes in the driver implementation

See :ref:`tutorial_driver` for a detailed walkthrough of creating a custom driver.

Supported Features Table
------------------------

The following table summarizes support across Qibolab drivers:

.. csv-table:: Supported features
    :header: "Feature", "RFSoC", "Qblox", "QM", "Emulator"
    :widths: 25, 10, 10, 10, 10

    "Arbitrary pulse sequence",     "✓","✓","✓","✓"
    "Arbitrary waveforms",          "✓","✓","✓","✓"
    "Multiplexed readout",          "✓","✓","✓","✓"
    "Hardware classification",      "✗","✓","✓","✗"
    "Fast reset",                   "dev","dev","dev","dev"
    "RTS frequency",                "✓","✓","✓","✓"
    "RTS amplitude",                "✓","✓","✓","✓"
    "RTS duration",                 "✓","✓","✓","✓"
    "RTS relative phase",           "✓","✓","✓","✓"
    "Hardware averaging",           "✓","✓","✓","✓"
    "Singleshot (no averaging)",    "✓","✓","✓","✓"
    "Integrated acquisition",       "✓","✓","✓","✓"
    "Classified acquisition",       "✓","✓","✓","✓"
    "Raw waveform acquisition",     "✓","✓","✓","✓"

Legend: ✓ = Supported, ✗ = Not supported, dev = Under development
