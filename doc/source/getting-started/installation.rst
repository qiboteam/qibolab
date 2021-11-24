Installation instructions
=========================

.. _installing-qibolab:

qibolab
^^^^^^^

Installing from source
""""""""""""""""""""""

The installation procedure presented in this section is useful when you have to
develop the code from source.

In order to install ``qibolab`` from source, you can simply clone the GitHub repository
with:

.. code-block::

      git clone https://github.com/qiboteam/qibolab.git
      cd qibolab
      pip install . # or pip install -e .

_______________________

.. _installing-tiiq:

tiiq
^^^^

In order to install the dependencies for the ``tiiq`` experiment 
use the following command:


.. code-block::

      pip install .[tiiq] # or pip install -e .[tiiq]

.. note::
      Currently there is an issue with the ``PyQt5-Qt5`` package if 
      you perform the installation of the ``tiiq`` dependencies on Windows
      in a conda environment with Python 3.8.

      In this case make sure to create a conda environment with a different Python
      version.


