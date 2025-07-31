How to add a new instrument in Qibolab?
=======================================

Currently, Qibolab supports various **controller** instruments:

* Quantum Machines

and the following **local oscillators**:

* Rhode&Schwartz
* Erasynth++

If you need to add a new driver, to support a new instruments in your setup, we encourage you to have a look at ``qibolab.instruments`` for complete examples.
In this section, anyway, a basic example will be given.

For clarity, we divide the instruments in two different groups: the **controllers** and the standard **instruments**,
where the controller is an instrument that can execute pulse sequences.
For example, a local oscillator is just an instrument, while Quantum Machines is a controller.

Add an instrument
-----------------

The base of an instrument is :class:`qibolab._core.instruments.abstract.Instrument`,
which is a pydantic ``Model``.
To accomodate different kind of instruments, a flexible interface is implemented
with two abstract methods (``connect()`` and ``disconnect()``) that are required
to be implemented in the child instrument.
Optionally, a ``setup()`` method can also be implemented to upload settings, such
as local oscillator frequency or power, to the instrument after connection.
If ``setup()`` is not implemented it will be an empty function.

In the execution of an experiment these functions are called sequentially, so
first a connection is established, the instrument is set up with the required
parameters, the instrument starts operating, then stops and gets disconnected.
Note that it's perfectly fine to leave the majority of these functions empty if
not needed.

Here, let's write a basic example of instrument whose job is to deliver a fixed bias for the duration of the experiment:

.. code-block::  python

    from typing import Optional

    # let's suppose that there is already available a base driver for connection
    # and control of the device, provided by the following library
    from proprietary_instruments import BiaserType, biaser_driver

    from qibolab.instruments.abstract import Instrument


    class Biaser(Instrument):
        """An instrument that delivers constand biases."""
        name: str
        address: str
        min_value: float = -1.0
        max_value: float = 1.0
        bias: float = 0.0
        device: Optional[BiaserType] = None


        def connect(self):
            """Check if a connection is avaiable."""
            if self.device is None:
                self.device = biaser_driver(self.address)
            self.device.on(self.bias)

        def disconnect(self):
            self.device.off(self.bias)
            self.device.disconnect()

        def setup(self):
            """Set biaser parameters."""
            self.device.set_range(self.min_value, self.max_value)


Add a controller
----------------

The controller is an instrument that has the additional method ``play``,
which allows it to execute arbitrary pulse sequences and perform sweeps.
Its abstract implementation can be found in :class:`qibolab._core.instruments.abstract.Controller`.

Let's see a minimal example:

.. code-block::  python

    from typing import Optional

    from proprietary_instruments import ControllerType, controller_driver

    from qibolab._core.components import Config
    from qibolab._core.execution_parameters import ExecutionParameters
    from qibolab._core.identifier import Result
    from qibolab._core.sequence import PulseSequence
    from qibolab._core.sweeper import ParallelSweepers
    from qibolab._core.instruments.abstract import Controller


    class MyController(Controller):

        def connect(self):
            if self.device is None:
                self.device = controller_driver(address)

        def disconnect(self):
            self.device.disconnect()

        def play(
                self,
                configs: dict[str, Config],
                sequences: list[PulseSequence],
                options: ExecutionParameters,
                sweepers: list[ParallelSweepers],
            ) -> dict[int, Result]:
            """Executes a PulseSequence."""
            if len(sweepers) > 0:
                raise NotImplementedError("MyController does not support sweeps.")

            if len(sequences) == 0:
                return {}
            elif len(sequences) == 1:
                sequence = sequences[0]
            else:
                sequence, _ = unroll_sequences(sequences, options.relaxation_time)

            # usually, some modification on the sequence, channel configs, or
            # parameters is needed so that the qibolab interface comply with the
            # interface of the device. Here these are assumed to be equal for simplicity.
            results = self.device.run_experiment(qubits, sequence, options)

            # also the results are, in qibolab, specific objects that need some kind
            # of conversion. Refer to the results section in the documentation.
            return results


Dummy platform
^^^^^^^^^^^^^^

In addition to the real instruments presented above, Qibolab provides the :class:`qibolab.instruments.DummyInstrument`.
This instrument represents a controller that returns random numbers of the proper shape when executing any pulse sequence.
This instrument is also part of the dummy platform which is defined in :py:mod:`qibolab._core.dummy` and can be initialized as

.. testcode::  python

    from qibolab import create_platform

    platform = create_platform("dummy")

This platform is equivalent to real platforms in terms of attributes and functions, but returns just random numbers.
It is useful for testing parts of the code that do not necessarily require access to an actual quantum hardware platform.
