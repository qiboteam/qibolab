Installation instructions
=========================

.. _installing-qibolab:

Qibolab
^^^^^^^
Installing with pip
"""""""""""""""""""

Installing with conda
"""""""""""""""""""""

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

.. _Platform:

Platforms supported
^^^^^^^^^^^^^^^^^^^


Qibolab already supports the following platforms:

* Tiiq
* IcarusQ

In the following sections we provide the specific installation instructions
to be ready to use Qibolab on the platform chosen.

Tiiq
""""

In order to install the dependencies for the ``Tiiq`` platform
use the following command:


.. code-block::

      pip install .[tiiq] # or pip install -e .[tiiq]

.. note::

      If you are working with the latest versions of MacOS, where the default shell is now ``zsh``,
      you will need to put ``.[tiiq]`` in quotes:

      .. code-block::

            pip install ".[tiiq]" # or pip install -e ".[tiiq]"


After that all you need to do to start using the ``Tiiq`` platform
is the following:

.. code-block:: python

      from qibolab import Platform

      platform = Platform("tiiq")

For more detailed applications see the :ref:`Code example <examples>` section.

IcarusQ
"""""""

In order to install the dependencies for the ``IcarusQ`` platform
use the following command:


.. code-block::

      pip install .[tiiq] # or pip install -e .[tiiq]

After that all you need to do to start using the ``IcarusQ`` platform
is the following:

.. code-block:: python

      from qibolab import Platform

      platform = Platform("icarusq")

For more detailed applications see the :ref:`Code example <examples>` section.
