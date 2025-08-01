.. _main_doc_instruments:

Instruments
===========

One the key features of Qibolab is its support for multiple different electronics.
A list of all the supported electronics follows:

Controllers (subclasses of :class:`qibolab._core.instruments.abstract.Controller`):
    - Dummy Instrument: :class:`qibolab.instruments.DummyInstrument`
    - Quantum Machines: :class:`qibolab.instruments.qm.QmController`

Other Instruments (subclasses of :class:`qibolab._core.instruments.abstract.Instrument`):
    - Erasynth++: :class:`qibolab.instruments.era.ERASynth`
    - RohseSchwarz SGS100A: :class:`qibolab.instruments.rohde_schwarz.SGS100A`

All instruments inherit the :class:`qibolab._core.instruments.abstract.Instrument` and implement methods for connecting and disconnecting.
:class:`qibolab._core.instruments.abstract.Controller` is a special case of instruments that provides the :class:`qibolab._core.instruments.abstract.execute`
method that deploys sequences on hardware.

Some more detail on the interal functionalities of instruments is given in :doc:`/tutorials/instrument`

The following is a table of the currently supported or not supported features (dev stands for `under development`):

.. csv-table:: Supported features
    :header: "Feature", "RFSoC", "Qblox", "QM"
    :widths: 25, 5, 5, 5, 5

    "Arbitrary pulse sequence",     "yes","yes","yes","yes"
    "Arbitrary waveforms",          "yes","yes","yes","yes"
    "Multiplexed readout",          "yes","yes","yes","yes"
    "Hardware classification",      "no","yes","yes","yes"
    "Fast reset",                   "dev","dev","dev","dev"
    "Device simulation",            "no","no","yes","dev"
    "RTS frequency",                "yes","yes","yes","yes"
    "RTS amplitude",                "yes","yes","yes","yes"
    "RTS duration",                 "yes","yes","yes","yes"
    "RTS relative phase",           "yes","yes","yes","yes"
    "RTS 2D any combination",       "yes","yes","yes","yes"
    "Sequence unrolling",           "dev","dev","dev","dev"
    "Hardware averaging",           "yes","yes","yes","yes"
    "Singleshot (no averaging)",    "yes","yes","yes","yes"
    "Integrated acquisition",       "yes","yes","yes","yes"
    "Classified acquisition",       "yes","yes","yes","yes"
    "Raw waveform acquisition",     "yes","yes","yes","yes"



Quantum Machines
^^^^^^^^^^^^^^^^

Tested with a cluster of nine `OPX+ <https://www.quantum-machines.co/products/opx/>`_ controllers, using QOP213 and QOP220.

Qibolab is communicating with the instruments using the `QUA <https://docs.quantum-machines.co/0.1/>`_ language, via the ``qm-qua`` and ``qualang-tools`` Python libraries.

.. _qrng:

Quantum Random Number Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the above instruments used for QPU control, Qibolab provides a driver
for sampling numbers from a quantum random number generator device (QRNG) in
:class:`qibolab.instruments.qrng.QRNG`.
This assumes that the device is connected to the host computer via a serial port.

The following script can be used to sample 1000 floats uniformly distributed in [0, 1]:

.. code::  python

    from qibolab.instruments.qrng import QRNG

    qrng = QRNG(address="/dev/ttyACM0")

    qrng.connect()

    samples = qrng.random(1000)

    qrng.disconnect()


The QRNG produces raw entropy which is converted to uniform distribution using an
exctraction algorithm. Two such algorithms are implemented

- :class:`qibolab.instruments.qrng.ShaExtrator`: default, based on SHA-256 hash algorithm,
- :class:`qibolab.instruments.qrng.ToeplitzExtractor`.

It is possible to switch extractor when instantiating the :class:`qibolab.instruments.qrng.QRNG` object:

.. code::  python

    from qibolab.instruments.qrng import QRNG, ToeplitzExtractor

    qrng = QRNG(address="/dev/ttyACM0", extractor=ToeplitzExtractor())


.. _main_doc_emulator:

Emulation of QPU platforms
---------------------------

Although Qibolab is mostly dedicated to providing hardware drivers for self-hosted quantum computing setups,
it is also possible to simulate the outcome of a pulse sequence with an emulator.
The emulator currently available is based on `QuTiP <https://qutip.org/>`_, the simulation is performed
by solving the master equation for a given Hamiltonian including dissipation using `mesolve <https://qutip.readthedocs.io/en/qutip-5.1.x/apidoc/solver.html#qutip.solver.mesolve.mesolve>`_.

With Qibolab it is currently possible to emulate a system of split-transmon qubits capacitively coupled. The Hamiltonian solved numerically in the case
of two transmon qubits is given by

.. math::

    \frac{H}{\hbar} =  \sum_{i=1}^2 \Big[ a^\dagger_i a_i \omega_i (\Phi_i) + \frac{\alpha_i}{2} a_i^\dagger a_i^\dagger a_i a_i - i \Omega_i(t) (a_i - a_i^\dagger) \Big] + g (a_1^\dagger a_2 + a_1 a_2^\dagger)

where :math:`a_i (a_i^\dagger)` are the destruction (creation) operators for the transmon  :math:`i`,
:math:`\omega_i` and :math:`\alpha_i / 2 \pi` are the frequency and the anharmoncity of the transmon  :math:`i`.
Each transmon is controlled with a drive term with a Rabi frequency :math:`\Omega_i(t)` and it is flux-tunable, meaning
that the frequency of the transmon can be changed by applying flux :math:`\Phi_i`

.. math::

    \omega_i(\Phi_i) = (\omega_i^{\text{max}} - \alpha_i)
    \sqrt[4]{d_i^2 + (1 - d_i^2)\cos^2\left( \pi k(\Phi_i - \Phi^{\text{sweetspot}}_i) \right)} + \alpha_i

where :math:`\omega_i^{\text{max}}` is the maximum frequency of the transmon, :math:`d_i` is the junctions asymmetry
and :math:`\Phi^{\text{sweetspot}}_i` is the flux value at which the transmon frequency is maximum
Currently neither drive or crosstalk effects are considered.
The coupling strength between the two transmons :math:`g` .

.. note::

    In most of the setups the sweetspot is identified by the offset value selected on the flux line connected to the qubit. Within the emulator
    it is possible to configure how to convert the offset value :math:`V` to the flux value using the entry ``voltage_to_flux``, which we denote with :math:`k` in the flux line configuration.
    The flux is computed from the offset value as

    .. math::

        \Phi = k \cdot V


The emulator supports also tunable based architecture, where the Hamiltonian is given by

.. math::

    \frac{H}{\hbar} =  \sum_{i=1,2,c} \Big[ a^\dagger_i a_i \omega_i + \frac{\alpha_i}{2} a_i^\dagger a_i^\dagger a_i a_i \Big] - \sum_{i=1,2} i \Omega_i(t) (a_i - a_i^\dagger)  + g_{12} (a_1^\dagger a_2 + a_1 a_2^\dagger) + g_{1c} (a_1^\dagger a_c + a_1 a_c^\dagger) + g_{2c} (a_2^\dagger a_c + a_2 a_c^\dagger) ,

where the index :math:`c` refers to the coupler.

The readout pulses parameters are ignored, given that the Hamiltonian doesn't include a resonator. The only information
used when the readout pulse is placed in the sequence which is necessary to determine for how long the system should be evolved.
The results retrieved by the emulator correspond to the time when the readout pulse is played.

Measurements are performed by measuring the probability of each transmon state available. In the case of two levels we return the probability
of finding the transmon in either :math:`\ket{0}` or :math:`\ket{1}`. When ``AveragingMode.SINGLESHOT`` is used samples are generated from the probabilities
computed previously. If ``AveragingMode.CYCLIC`` the following weighted average is returned

.. math::

    \mu = \sum_{i=0}^{N} i  p_i

where :math:`p_i` is the probability corresponding to state :math:`\ket{i}`, and :math:`N` are the transmon levels available.

The emulator supports ``AcquisitionType.DISCRIMINATION``. We also provide a way of retrieving information with ``AcquisitionType.INTEGRATION``
by encoding into the :math:`I` component the probabilities and while the :math:`Q` component is set at 0.
We add a Gaussian noise both on :math:`I` and :math:`Q`.
This should be enough to get some meaningful results by computing the magnitude of the signal :math:`\sqrt{I^2 + Q^2}`.

Example of platforms using the emulator are available `here <https://https://github.com/qiboteam/qibolab/tree/emulator-tests/tests/instruments/emulator/platforms/>`_.
