.. _installing-qibolab:

Installation
============

Qibolab requires Python 3.11, 3.12, or 3.13. The base package provides the pulse
and platform APIs and a built-in dummy platform, so you can work through the
:doc:`first experiment <experiment>` without laboratory equipment or Qibo.
Install it in a virtual environment:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install qibolab

On Windows PowerShell, activate the environment with
``.venv\Scripts\Activate.ps1`` instead. Using ``python -m pip`` ensures that
packages are installed for the interpreter you will run.

Choose the optional features you need
-------------------------------------

Circuit execution through Qibo is optional. Install the ``backend`` extra to
use Qibolab as a Qibo backend:

.. code-block:: bash

    python -m pip install "qibolab[backend]"

Numerical emulation has a separate set of dependencies:

.. code-block:: bash

    python -m pip install "qibolab[emulator]"

You can combine extras as ``"qibolab[backend,emulator]"``. Installing emulation
dependencies does not automatically create or calibrate an emulated platform;
see the :doc:`emulation guide <../main-documentation/emulator>` for the
distinction between a dummy platform and a numerical model.

For real hardware, install the integration dependencies required by the
platform supplied by your laboratory. They are separate from the base
package, and their installation and device-specific setup are outside the
scope of this documentation. Loading a platform definition may import those
dependencies even before you connect to equipment.

Install from source
--------------------

For development, clone the repository and create an editable installation:

.. code-block:: bash

    git clone https://github.com/qiboteam/qibolab.git
    cd qibolab
    python -m pip install -e .

The repository also supports ``uv``. ``uv sync`` installs the project and its
default development dependency group; request other groups or extras explicitly:

.. code-block:: bash

    uv sync --group docs --extra backend --extra emulator
    uv run make -C doc html

The generated documentation is written to ``doc/build/html``. To execute the
documentation's testable examples independently of a cached Sphinx environment:

.. code-block:: bash

    uv run make -C doc doctest SPHINXOPTS='--fresh-env'

Finding laboratory platforms
-----------------------------

An installed Qibolab package and a configured laboratory platform are different
things. Except for ``dummy``, ``create_platform("name")`` looks for a local
platform definition using ``QIBOLAB_PLATFORMS``. Neither installing the base
package nor installing an extra supplies your laboratory's wiring or calibration
parameters. Follow :doc:`../tutorials/storage` to configure platform discovery,
and :doc:`../tutorials/lab` if you are assembling a platform yourself.
