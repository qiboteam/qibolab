.. title::
   Qibolab


What is Qibolab?
================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7973899.svg
   :target: https://doi.org/10.5281/zenodo.7973899

Qibolab is the dedicated `Qibo <https://github.com/qiboteam/qibo>`_ backend for
quantum hardware control. This module automates the implementation of quantum
circuits on quantum hardware. Qibolab includes:

1. *Platform API*: support custom allocation of quantum hardware platforms / lab setup.
2. *Drivers*: supports commercial and open-source firmware for hardware control.
3. *Arbitrary pulse API*: provide a library of custom pulses for execution through instruments.
4. *Transpiler*: compiles quantum circuits into pulse sequences matching chip topology.
5. *Quantum Circuit Deployment*: seamlessly deploys quantum circuit models on
   quantum hardware.

Components
----------

.. image:: platform_object.svg

Key features
------------

* Deploy Qibo models on quantum hardware easily.
* Create custom experimental drivers for custom lab setup.
* Support multiple heterogeneous platforms.
* Use existing calibration procedures for experimentalists.

Contents
========

.. toctree::
    :maxdepth: 2
    :caption: Introduction

    getting-started/index
    tutorials/index

.. toctree::
    :maxdepth: 2
    :caption: Main documentation

    main-documentation/index
    api-reference/index
    developer-guides/index

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
==================

* :ref:`genindex`
* :ref:`search`
