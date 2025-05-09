.. title::
   Qibolab


What is Qibolab?
================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7973899.svg
   :target: https://doi.org/10.5281/zenodo.7973899

Qibolab is the dedicated `Qibo <https://github.com/qiboteam/qibo>`_ package for
quantum hardware control. This module automates the implementation of quantum
circuits on quantum hardware.

Structure
^^^^^^^^^

Qibolab's architecture comprises two primary components:


- the :ref:`Platform API <main_doc_platform>`, which facilitates the custom allocation
  of quantum hardware platforms and laboratory setups, and
- its :ref:`Experiment API <main_doc_experiment>`, which provides the tools to define
  arbitrary experiments, based on pulse sequences, for execution on the configured
  platforms.

Platforms’ definition involve describing the arrangement of physical devices using
Qibolab's abstractions. This is achieved through provided :ref:`drivers
<main_doc_instruments>`, which offer support for both commercial and open-source
firmware for hardware control.

In addition to pulse execution, Qibolab platforms function as backends for :ref:`quantum
circuit deployment <tutorials_circuits>` on hardware. This functionality is enabled by
an integrated circuit :ref:`compiler <main_doc_compiler>`, which translates quantum
circuits into pulse sequences.


Qibolab is designed to be used in conjunction with `Qibocal
<https://github.com/qiboteam/qibocal>`_, which supplies a comprehensive suite of
calibration procedures for any Qibolab-based platform.


Contents
^^^^^^^^

.. toctree::
    :maxdepth: 2
    :caption: Getting started

    getting-started/installation
    getting-started/experiment

.. toctree::
    :maxdepth: 2
    :caption: Core components

    main-documentation/platform
    main-documentation/experiment
    main-documentation/drivers
    main-documentation/compiler

.. toctree::
    :maxdepth: 2
    :caption: Tutorials

    tutorials/lab
    tutorials/pulses
    tutorials/circuits
    tutorials/calibration
    tutorials/instrument
    tutorials/emulator
    tutorials/storage

.. toctree::
    :maxdepth: 1
    :caption: Reference

    api-reference/qibolab

.. toctree::
    :maxdepth: 2
    :caption: Contributing

    Developer guides <https://qibo.science/qibo/stable/developer-guides/index.html>

.. toctree::
    :maxdepth: 2
    :caption: Appendix

    Publications <https://qibo.science/qibo/stable/appendix/citing-qibo.html>

.. toctree::
    :maxdepth: 1
    :caption: Documentation links

    Qibo docs <https://qibo.science/qibo/stable/>
    Qibolab docs <https://qibo.science/qibolab/stable/>
    Qibocal docs <https://qibo.science/qibocal/stable/>
    Qibosoq docs <https://qibo.science/qibosoq/stable/>


Indices and tables
^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`search`
