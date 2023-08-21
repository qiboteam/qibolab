How to connect Qibolab to your lab?
===================================

In this section we will show how to let Qibolab communicate with your lab's
instruments and run an experiment.

The main required object, in this case, is the `Platform`. A Platform is defined
as a QPU (quantum processing unit with one or more qubits) controlled by one ore
more instruments.

How to define a platform for a self-hosted QPU?
-----------------------------------------------

The :class:`qibolab.platform.Platform` object holds all the information required
to execute programs, and in particular :class:`qibolab.pulses.PulseSequence` in
a real QPU. It is comprised by different objects that contain information about
the qubit characterization and connectivity, the native gates and the lab's
instrumentation.

The following cell shows how to define a single qubit platform from scratch,
using different Qibolab primitives.

.. code-block::  python

    from qibolab import Platform
    from qibolab.qubits import Qubit
    from qibolab.pulses import PulseType
    from qibolab.channels import ChannelMap, Channel
    from qibolab.native import NativePulse, SingleQubitNatives
    from qibolab.instruments.dummy import DummyInstrument


    def create():
        # Create a controller instrument
        instrument = DummyInstrument("my_instrument", "0.0.0.0:0")

        # Create channel objects and assign to them the controller ports
        channels = ChannelMap()
        channels |= Channel("ch1out", port=instrument["o1"])
        channels |= Channel("ch2", port=instrument["o2"])
        channels |= Channel("ch1in", port=instrument["i1"])

        # create the qubit object
        qubit = Qubit(0)

        # assign native gates to the qubit
        qubit.native_gates = SingleQubitNatives(
            RX=NativePulse(
                name="RX",
                duration=40,
                amplitude=0.05,
                shape="Gaussian(5)",
                pulse_type=PulseType.DRIVE,
                qubit=qubit,
                frequency=int(4.5e9)
            ),
            MZ=NativePulse(
                name="MZ",
                duration=1000,
                amplitude=0.005,
                shape="Rectangular()",
                pulse_type=PulseType.READOUT,
                qubit=qubit,
                frequency=int(7e9)
            )
        )

        # assign channels to the qubit
        qubit.readout = channels["ch1out"]
        qubit.feedback = channels["ch1in"]
        qubit.drive = channels["ch2"]

        # create dictionaries of the different objects
        qubits = {qubit.name: qubit}
        pairs = {} # empty as for single qubit we have no qubit pairs
        instruments = {instrument.name: instrument}

        # allocate and return Platform object
        return Platform("my_platform", qubits, pairs, instruments, resonator_type="3D")


This code creates a platform with a single qubit that is controlled by the
:class:`qibolab.instruments.dummy.DummyInstrument`. In real applications, if
Qibolab provides drivers for the instruments in the lab, these can be directly
used in place of the ``DummyInstrument`` above, otherwise new drivers need to be
coded following the abstract :class:`qibolab.instruments.abstract.Instrument`
interface.

Furthermore, above we defined three channels that connect the qubit to the
control instrument and we assigned two native gates to the qubit. In this
example we neglected or characterization parameters associated to the qubit.
These can be passed when defining the :class:`qibolab.qubits.Qubit` objects.

When the QPU contains more than one qubit, some of the qubits are connected so
that two-qubit gates can be applied. For such connected pairs of qubits one
needs to additionally define :class:`qibolab.qubits.QubitPair` objects, which
hold the parameters of the two-qubit gates.

.. code-block::  python

    from qibolab.qubits import Qubit, QubitPair
    from qibolab.pulses import PulseType
    from qibolab.native import (
        NativePulse,
        NativeSequence,
        SingleQubitNatives,
        TwoQubitNatives,
    )

    # create the qubit objects
    qubit0 = Qubit(0)
    qubit1 = Qubit(1)

    # assign single-qubit native gates to each qubit
    qubit0.native_gates = SingleQubitNatives(
        RX=NativePulse(
            name="RX",
            duration=40,
            amplitude=0.05,
            shape="Gaussian(5)",
            pulse_type=PulseType.DRIVE,
            qubit=qubit0,
            frequency=int(4.7e9),
        ),
        MZ=NativePulse(
            name="MZ",
            duration=1000,
            amplitude=0.005,
            shape="Rectangular()",
            pulse_type=PulseType.READOUT,
            qubit=qubit0,
            frequency=int(7e9),
        ),
    )
    qubit1.native_gates = SingleQubitNatives(
        RX=NativePulse(
            name="RX",
            duration=40,
            amplitude=0.05,
            shape="Gaussian(5)",
            pulse_type=PulseType.DRIVE,
            qubit=qubit1,
            frequency=int(5.1e9),
        ),
        MZ=NativePulse(
            name="MZ",
            duration=1000,
            amplitude=0.005,
            shape="Rectangular()",
            pulse_type=PulseType.READOUT,
            qubit=qubit1,
            frequency=int(7.5e9),
        ),
    )

    # define the pair of qubits
    pair = QubitPair(qubit0, qubit1)
    pair.native_gates = TwoQubitNatives(
        CZ=NativeSequence(
            name="CZ",
            pulses=[
                NativePulse(
                    name="CZ1",
                    duration=30,
                    amplitude=0.005,
                    shape="Rectangular()",
                    pulse_type=PulseType.FLUX,
                    qubit=qubit1,
                )
            ],
        )
    )



The platform automatically creates the connectivity graph of the given chip
using the dictionary of :class:`qibolab.qubits.QubitPair` objects.

Registering platforms
^^^^^^^^^^^^^^^^^^^^^

The ``create()`` function defined in the above example can be called or imported
directly in any Python script. Alternatively, it is also possible to make the
platform available as

.. code-block::  python

    from qibolab import Platform

    # Define platform and load specific runcard
    platform = Platform("my_platform")


To do so, ``create()`` needs to be saved in a module called ``my_platform.py``
and the environment flag ``QIBOLAB_PLATFORMS`` needs to point to the directory
that contains this module. Examples of advanced platforms are available at `this
repository <https://github.com/qiboteam/qibolab_platforms_qrc>`_.

.. _using_runcards:

Using runcards
^^^^^^^^^^^^^^

Operating a QPU requires calibrating a set of parameters, the number of which
increases with the number of qubits. Hardcoding such parameters in the
``create()`` function, as shown in the above examples, is not scalable. However,
since ``create()`` is part of a Python module, is is possible to load parameters
from an external file or database.

Qibolab provides some utility functions, accessible through
:py:mod:`qibolab.serialize`, for loading calibration parameters stored in a YAML
file with a specific format. We call such file a runcard. Here is a runcard for
a two-qubit system:

.. code-block::  yaml

    nqubits: 2

    qubits: [0, 1]

    settings:
        nshots: 1024
        sampling_rate: 1000000000
        relaxation_time: 50_000

    topology: [[0, 1]]

    native_gates:
        single_qubit:
            0: # qubit number
                RX:
                    duration: 40
                    amplitude: 0.0484
                    frequency: 4_855_663_000
                    shape: Drag(5, -0.02)
                    type: qd # qubit drive
                    start: 0
                    phase: 0
                MZ:
                    duration: 620
                    amplitude: 0.003575
                    frequency: 7_453_265_000
                    shape: Rectangular()
                    type: ro # readout
                    start: 0
                    phase: 0
            1: # qubit number
                RX:
                    duration: 40
                    amplitude: 0.05682
                    frequency: 5_800_563_000
                    shape: Drag(5, -0.04)
                    type: qd # qubit drive
                    start: 0
                    phase: 0
                MZ:
                    duration: 960
                    amplitude: 0.00325
                    frequency: 7_655_107_000
                    shape: Rectangular()
                    type: ro # readout
                    start: 0
                    phase: 0

        two_qubit:
            0-1:
                CZ:
                - duration: 30
                  amplitude: 0.055
                  shape: Rectangular()
                  qubit: 1
                  relative_start: 0
                  type: qf
                - type: virtual_z
                  phase: -1.5707963267948966
                  qubit: 0
                - type: virtual_z
                  phase: -1.5707963267948966
                  qubit: 1

    characterization:
        single_qubit:
            0:
                readout_frequency: 7_453_265_000
                drive_frequency: 4_855_663_000
                T1: 0.0
                T2: 0.0
                sweetspot: -0.047
                # parameters for single shot classification
                threshold: 0.00028502261712637096
                iq_angle: 1.283105298787488
            1:
                readout_frequency: 7_655_107_000
                drive_frequency: 5_800_563_000
                T1: 0.0
                T2: 0.0
                sweetspot: -0.045
                # parameters for single shot classification
                threshold: 0.0002694329123116206
                iq_angle: 4.912447775569025


This file contains different sections: ``qubits`` is a list with the qubit
names, ``settings`` defines default execution parameters, ``topology`` defines
the qubit connectivity (qubit pairs), ``native_gates`` specifies the calibrated
pulse parameters for implementing single and two-qubit gates and
``characterization`` provides the physical parameters associated to each qubit.
Note that such parameters may slightly differ depending on the QPU architecture,
however the pulses under ``native_gates`` should comply with the
:class:`qibolab.pulses.Pulse` API and the parameters under ``characterization``
should be a subset of :class:`qibolab.qubits.Qubit` attributes.

Providing the above runcard is not sufficient to instantiate a
:class:`qibolab.platform.Platform`. This should still be done using a
``create()`` method, however this is significantly simplified by
``qibolab.serialize``. Here is the ``create()`` method that loads the parameters of
the above runcard:

.. code-block::  python

    from pathlib import Path
    from qibolab import Platform
    from qibolab.channels import ChannelMap, Channel
    from qibolab.serialize import load_runcard, load_qubits, load_settings
    from qibolab.instruments.dummy import DummyInstrument


    def create():
        # Create a controller instrument
        instrument = DummyInstrument("my_instrument", "0.0.0.0:0")

        # Create channel objects and assign to them the controller ports
        channels = ChannelMap()
        channels |= Channel("ch1out", port=instrument["o1"])
        channels |= Channel("ch2", port=instrument["o2"])
        channels |= Channel("ch3", port=instrument["o3"])
        channels |= Channel("ch1in", port=instrument["i1"])

        # create ``Qubit`` and ``QubitPair`` objects by loading the runcard
        runcard = load_runcard(Path(__file__).parent / "my_platform.yml")
        qubits, pairs = load_qubits(runcard)

        # assign channels to the qubit
        for q in range(2):
            qubits[q].readout = channels["ch1out"]
            qubits[q].feedback = channels["ch1in"]
            qubits[q].drive = channels[f"ch{q + 2}"]

        # create dictionary of instruments
        instruments = {instrument.name: instrument}
        # load ``settings`` from the runcard
        settings = load_settings(runcard)
        return Platform(
            "my_platform", qubits, pairs, instruments, settings, resonator_type="2D"
        )

Note that this assumes that the runcard is saved as ``my_platform.yml`` in the
same directory with the Python file that contains ``create()``.
