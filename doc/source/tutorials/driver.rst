.. _tutorial_driver:

Creating a Custom Instrument Driver
====================================

This tutorial covers implementing a custom instrument driver for Qibolab.

Overview
--------

An instrument driver is a Python class that:

1. Inherits from :class:`._core.instruments.abstract.Instrument`
2. Implements connection/disconnection
3. (For controllers) Implements experiment execution

When to Create a Driver
-----------------------

Create a driver when:

- You have custom hardware not supported by existing drivers
- You want to add platform-specific optimizations
- You need to support special features or configurations

Minimal Instrument Driver
--------------------------

Here's a minimal non-controller instrument (e.g., LO):

.. code-block:: python

    from qibolab._core.instruments.abstract import Instrument, InstrumentSettings
    from pydantic import Field


    class MyLocalOscillator(Instrument):
        """Custom local oscillator driver."""

        address: str
        settings: InstrumentSettings | None = None
        frequency: float = Field(default=5e9, description="LO frequency")

        def connect(self):
            """Establish connection to LO."""
            print(f"Connecting to LO at {self.address}")
            # Implement actual connection logic
            # e.g., GPIB, Ethernet, etc.

        def disconnect(self):
            """Close connection to LO."""
            print(f"Disconnecting from LO at {self.address}")

        def setup(self, **kwargs):
            """Configure LO settings."""
            print(f"Setting LO frequency to {self.frequency/1e9} GHz")
            # Implement hardware configuration

Using Your Driver
~~~~~~~~~~~~~~~~~~

Use it in a platform:

.. code-block:: python

    from qibolab._core.platform import Platform, Hardware
    from qibolab._core.components import IqChannel

    lo = MyLocalOscillator(address="192.168.1.1", frequency=5e9)

    hardware = Hardware(
        instruments={"lo": lo},
        qubits={},
        couplers={},
    )

    platform = Platform(
        name="my_platform",
        parameters=parameters,
        instruments={"lo": lo},
        qubits={},
    )

Minimal Controller Driver
--------------------------

A controller must implement the ``play()`` method:

.. code-block:: python

    from qibolab._core.instruments.abstract import Controller
    from qibolab._core.components import Channel, Config
    from qibolab._core.execution_parameters import ExecutionParameters
    from qibolab._core.sequence import PulseSequence
    from qibolab._core.sweeper import ParallelSweepers
    from qibolab._core.identifier import Result
    import numpy as np
    from pydantic import Field


    class MyPulseController(Controller):
        """Custom pulse controller."""

        address: str
        sampling_rate_value: float = 5.0  # GSps
        channels: dict = Field(default_factory=dict)

        @property
        def sampling_rate(self) -> float:
            """Return sampling rate in GSps."""
            return self.sampling_rate_value

        def connect(self):
            """Connect to hardware."""
            print(f"Connecting to controller at {self.address}")

        def disconnect(self):
            """Disconnect from hardware."""
            print(f"Disconnecting from controller at {self.address}")

        def play(
            self,
            configs: dict[str, Config],
            sequences: list[PulseSequence],
            options: ExecutionParameters,
            sweepers: list[ParallelSweepers],
        ) -> dict[int, Result]:
            """Execute experiment."""

            # Step 1: Compile sequences to hardware instructions
            hardware_program = self._compile_sequences(sequences, configs, sweepers)

            # Step 2: Upload program to hardware
            self._upload_program(hardware_program)

            # Step 3: Configure acquisition
            self._setup_acquisition(options)

            # Step 4: Run experiment
            raw_results = self._run_experiment(options)

            # Step 5: Process and return results
            return self._process_results(raw_results, options, sequences)

        def _compile_sequences(self, sequences, configs, sweepers):
            """Convert Qibolab sequences to hardware format."""
            # Implementation depends on your hardware instruction set
            # Example: convert pulses to waveform definitions, timing, etc.
            return {}

        def _upload_program(self, program):
            """Upload compiled program to hardware."""
            print("Uploading program to hardware")

        def _setup_acquisition(self, options):
            """Configure acquisition settings."""
            print(f"Setting up acquisition: {options.acquisition_type}")

        def _run_experiment(self, options):
            """Run experiment on hardware and retrieve data."""
            print(f"Running experiment with {options.nshots} shots")
            # Retrieve raw results from hardware
            return {}

        def _process_results(self, raw, options, sequences):
            """Process raw hardware results into Qibolab format."""
            results = {}
            for seq in sequences:
                for ch, pulse in seq.acquisitions:
                    results[pulse.id] = np.zeros(options.results_shape([]))
            return results

Advanced: Custom Channels and Configs
--------------------------------------

Define custom channel types for your hardware:

.. code-block:: python

    from qibolab._core.components import IqChannel


    class MyCustomChannel(IqChannel):
        """Platform-specific channel with extra parameters."""

        custom_param: str = "default"
        hardware_id: int = 0

Use custom configs to store platform-specific settings:

.. code-block:: python

    from qibolab._core.components.configs import IqConfig


    class MyCustomConfig(IqConfig):
        """Platform-specific IQ config."""

        calibration_date: str = ""
        temperature: float = 0.0
        extra_param: float = 0.0

Advanced: Real-Time Sweepers
-----------------------------

Implement hardware sweeper support for efficiency:

.. code-block:: python

    def play(self, configs, sequences, options, sweepers):
        # Check if sweepers are supported by hardware
        if sweepers and not self._supports_sweepers(sweepers):
            # Unroll sweepers: execute sequences in a loop
            return self._unroll_sweepers(sequences, configs, options, sweepers)

        # Compile with native sweeper support
        program = self._compile_with_sweepers(sequences, configs, sweepers)
        # ... rest of execution


    def _supports_sweepers(self, sweepers):
        """Check if all sweepers are supported."""
        # Implement hardware-specific checks
        return True


    def _unroll_sweepers(self, sequences, configs, options, sweepers):
        """Unroll sweepers as nested loops."""
        # Manually iterate through sweep points
        pass

Error Handling
--------------

Implement robust error handling:

.. code-block:: python

    def play(self, configs, sequences, options, sweepers):
        try:
            # Validate inputs
            self._validate_sequences(sequences)
            self._validate_configs(configs)

            # Execute
            program = self._compile_sequences(sequences, configs, sweepers)
            self._upload_program(program)
            results = self._run_experiment(options)

            return self._process_results(results, options, sequences)

        except ConnectionError as e:
            raise InstrumentException(self, f"Connection failed: {e}")
        except ValueError as e:
            raise InstrumentException(self, f"Invalid configuration: {e}")
        except Exception as e:
            raise InstrumentException(self, f"Execution failed: {e}")


    def _validate_sequences(self, sequences):
        """Validate sequences before compilation."""
        for seq in sequences:
            for ch, pulse in seq:
                if ch not in self.channels:
                    raise ValueError(f"Channel {ch} not found in controller")

Testing Your Driver
-------------------

Test your driver without hardware:

.. code-block:: python

    def test_my_controller():
        from qibolab._core.components import IqChannel
        from qibolab import Pulse, PulseSequence, Rectangular

        # Create controller
        controller = MyPulseController(address="dummy")
        controller.channels = {
            "drive": IqChannel(path=(0,)),
        }

        # Create test sequence
        pulse = Pulse(duration=40, amplitude=0.5, envelope=Rectangular())
        sequence = PulseSequence([("drive", pulse)])

        # Create test options
        from qibolab import AcquisitionType, AveragingMode

        options = ExecutionParameters(
            nshots=100,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        )

        # Test play() method
        results = controller.play(
            configs={},
            sequences=[sequence],
            options=options,
            sweepers=[],
        )

        # Verify results
        assert results is not None
        print("Test passed!")


    if __name__ == "__main__":
        test_my_controller()

Testing Integration
~~~~~~~~~~~~~~~~~~~

Test your driver with a full platform:

.. code-block:: python

    def test_platform_with_custom_driver():
        # Create platform with your driver
        from qibolab._core.platform import Platform, Hardware
        from qibolab._core.qubits import Qubit

        controller = MyPulseController(address="dummy")
        hardware = Hardware(
            instruments={"controller": controller},
            qubits={0: Qubit(drive="drive")},
        )

        platform = Platform(
            name="test",
            parameters=parameters,
            instruments={"controller": controller},
            qubits={0: Qubit(drive="drive")},
        )

        # Test workflow
        platform.connect()
        sequence = platform.natives.single_qubit[0].RX()
        results = platform.execute([sequence])
        platform.disconnect()

        print("Platform integration test passed!")

Best Practices
--------------

1. **Validate inputs**: Check sequences, configs, and options early
2. **Handle hardware timeouts**: Set reasonable timeouts for communication
3. **Cache compiled programs**: Avoid recompilation when possible
4. **Log operations**: Use Python's logging for debugging
5. **Document your driver**: Clear docstrings for configuration parameters
6. **Test offline**: Provide a dummy mode for testing without hardware
7. **Follow Pydantic patterns**: Use Pydantic for configuration validation

Example: Dummy Controller
--------------------------

See the implementation in ``src/qibolab/_core/instruments/dummy.py`` for a
complete working controller example.

Next Steps
----------

- See :ref:`main_doc_driver_api` for complete API reference
- Explore existing drivers in ``src/qibolab/_core/instruments/``
- Implement custom channel and config types
- Add hardware-specific optimizations
- Write comprehensive tests for your driver
