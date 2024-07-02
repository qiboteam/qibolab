How to add a new instrument in Qibolab?
=======================================

Currently, Qibolab support various instruments:
as **controller**:

* Quantum Machines
* QBlox
* Zurich instruments
* Xilinx RFSoCs

and as **local oscillators**:

* Rhode&Schwartz
* Erasynth++

If you need to add a new driver, to support a new instruments in your setup, we encourage you to have a look at ``qibolab.instruments`` for complete examples.
In this section, anyway, a basic example will be given.

For clarity, we divide the instruments in two different groups: the **controllers** and the standard **instruments**, where the controller is an instrument that can execute pulse sequences.
For example, a local oscillator is just an instrument, while QBlox is a controller.

Add an instrument
-----------------

The base of an instrument is :class:`qibolab.instruments.abstract.Instrument`.
To accomodate different kind of instruments, a flexible interface is implemented
with four abstract methods that are required to be implemented in the child
instrument:

* ``connect()``
* ``setup()``
* ``start()``
* ``stop()``
* ``disconnect()``

In the execution of an experiment these functions are called sequentially, so
first a connection is established, the instrument is set up with the required
parameters, the instrument starts operating, then stops and gets disconnected.
Note that it's perfectly fine to leave the majority of these functions empty if
not needed.

Moreover, it's important call the ``super.__init__(self, name, address)`` since
it initialize the folders eventually required to store temporary files.

A good example of a instrument driver is the
:class:`qibolab.instruments.rohde_schwarz.SGS100A` driver.

Here, let's write a basic example of instrument whose job is to deliver a fixed bias for the duration of the experiment:

.. literalinclude:: ./includes/instrument/instrument0.py

Add a controller
----------------

The controller is an instrument that has some additional methods, its abstract
implementation can be found in :class:`qibolab.instruments.abstract.Controller`.

The additional methods required are:

* ``play()``
* ``play_sequences()``
* ``sweep()``

The simplest real example is the RFSoCs driver in
:class:`qibolab.instruments.rfsoc.driver.RFSoC`, but still the code is much more
complex than the local oscillator ones.

Let's see a minimal example:

.. literalinclude:: ./includes/instrument/instrument1.py

As we saw in :doc:`lab`, to instantiate a platform at some point you have to
write something like this:

.. testcode:: python

    from qibolab.channels import Channel, ChannelMap
    from qibolab.instruments.dummy import DummyInstrument

    instrument = DummyInstrument("my_instrument", "0.0.0.0:0")
    channels = ChannelMap()
    channels |= Channel("ch1out", port=instrument.ports("o1"))


The interesting part of this section is the ``port`` parameter that works as an
attribute of the controller. A :class:`qibolab.instruments.port.Port` object
describes the physical connections that a device may have. A Controller has, by
default, ports characterized just by ``port_name`` (see also
:class:`qibolab.instruments.abstract.Controller`), but different devices may
need to add attributes and methods to the ports. This can be done by defining in
the new controller a new port type. See, for example, the already implemented
ports:

* :class:`qibolab.instruments.rfsoc.driver.RFSoCPort`
* :class:`qibolab.instruments.qm.ports.QMPort`
* :class:`qibolab.instruments.zhinst.ZhPort`
* :class:`qibolab.instruments.qblox.port`
