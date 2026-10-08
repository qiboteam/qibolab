.. title:: Qibolab

Qibolab: from experiments to quantum hardware
=============================================

Qibolab provides a Python interface for defining pulse experiments and executing
them on quantum platforms. A platform brings together the laboratory's hardware
arrangement, its configuration, and the native pulse operations used to control
the device. An experiment describes what to play, what to vary, and what data
to acquire.

You can use the pulse API independently of
`Qibo <https://qibo.science/qibo/stable/>`_. With the optional Qibo backend,
Qibolab also translates quantum circuits into native pulse sequences and returns
circuit measurement results. For calibration protocols and analysis, it works
with `Qibocal <https://qibo.science/qibocal/stable/>`_.

Start with :doc:`getting-started/installation` and
:doc:`getting-started/experiment`. The first experiment runs without hardware
using a dummy platform, which returns random data with the expected layout.
It is a way to learn the API, not a quantum simulator.

How to use these docs
---------------------

The conceptual guides explain the platform and experiment models, result
layout, and the path from a circuit to pulses. The tutorials build on those
ideas with concrete tasks: constructing sequences, sweeping parameters,
assembling platforms, and saving calibrations. The API reference is for looking
up signatures and fields after you understand the workflow.

If you already have a laboratory platform, focus on :doc:`tutorials/pulses`
and :doc:`tutorials/sweeps`. If you are integrating a new setup, begin with
:doc:`main-documentation/platform`, then :doc:`tutorials/lab` and
:doc:`tutorials/storage`. Circuit users can go directly to
:doc:`tutorials/circuits` after installing the backend extra.

These pages describe Qibolab's common interfaces. Individual instrument
drivers, their implementation, and device-specific setup are intentionally
outside the scope of this documentation.

.. toctree::
    :maxdepth: 2
    :caption: Getting started

    getting-started/installation
    getting-started/experiment

.. toctree::
    :maxdepth: 2
    :caption: Conceptual guides

    main-documentation/platform
    main-documentation/experiment
    main-documentation/circuits
    main-documentation/compiler
    main-documentation/emulator

.. toctree::
    :maxdepth: 2
    :caption: Tutorials

    tutorials/pulses
    tutorials/sweeps
    tutorials/calibration
    tutorials/lab
    tutorials/storage
    tutorials/circuits
    tutorials/emulator

.. toctree::
    :maxdepth: 2
    :caption: API reference

    api-reference/public
    api-reference/platform
    api-reference/backend

.. toctree::
    :maxdepth: 1
    :caption: Further reading

    references
    Qibo documentation <https://qibo.science/qibo/stable/>
    Qibocal documentation <https://qibo.science/qibocal/stable/>
    Developer guides <https://qibo.science/qibo/stable/developer-guides/index.html>
    Citation policy <https://qibo.science/qibo/stable/appendix/citing-qibo.html>

Indices
-------

* :ref:`genindex`
* :ref:`search`
