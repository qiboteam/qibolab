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

In order to install the dependencies for the ``tiiq`` platform
use the following command:


.. code-block::

      pip install .[tiiq] # or pip install -e .[tiiq]

.. note::

      If you are working with the latest versions of MacOS, where the default shell is now ``zsh``,
      you will need to put ``.[tiiq]`` in quotes:

      .. code-block::

            pip install ".[tiiq]" # or pip install -e ".[tiiq]"


