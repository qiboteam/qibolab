Platforms storage
=================

To define a platform the user needs to provide a folder with the following structure:

.. code-block:: bash

    my_platform/
        platform.py
        parameters.json

where ``platform.py`` contains instruments information and ``parameters.json`` includes calibration parameters.

Setting up the environment
--------------------------

After defining the platform, we must instruct ``qibolab`` of the location of the platform(s).
We need to define the path that contains platform folders.
This can be done using an environment variable:
for Unix based systems:

.. code-block:: bash

    export QIBOLAB_PLATFORMS=<path-platform-folders>

for Windows:

.. code-block:: bash

    $env:QIBOLAB_PLATFORMS="<path-to-platform-folders>"

To avoid having to repeat this export command for every session, this line can be added to the ``.bashrc`` file (or alternatives such as ``.zshrc``).
