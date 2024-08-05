Calibration experiments
=======================

Let's see some examples of the typical experiments needed to calibrate and
characterize a qubit.

.. note::
   This is just for demonstration purposes! In the `Qibo <https://qibo.science/qibo/stable/>`_ framework these experiments are already coded and available in the `Qibocal API <https://qibo.science/qibocal/stable/>`_.

Let's consider a platform called `single_qubit` with, as expected, a single
qubit.

Resonator spectroscopy
----------------------

The first experiment we conduct is a resonator spectroscopy. The experiment is
as follows:

1. A measurement pulse (pulse on the readout line, followed by an acquisition)
    is fired at a specific frequency.
2. We repeat point 1 for different frequencies.
3. We plot the acquired amplitudes, identifying the peak/deep value as the
   resonator frequency.

We start by initializing the platform, that reads the information written in the
respective runcard, a sequence composed of only a measurement and a sweeper
around the pre-defined frequency.

.. literalinclude:: ./includes/calibration/calibration0.py

.. image:: resonator_spectroscopy_light.svg
   :class: only-light
.. image:: resonator_spectroscopy_dark.svg
   :class: only-dark

Qubit spectroscopy
------------------

For a qubit spectroscopy experiment, the procedure is almost identical. A
typical qubit spectroscopy experiment is as follows:

1. A first pulse is sent to the drive line, in order to excite the qubit. Since
   the qubit parameters are not known, this is typically a very long pulse (2
   microseconds) at low amplitude.
2. A measurement, tuned with resonator spectroscopy, is performed.
3. We repeat point 1 for different frequencies.
4. We plot the acquired amplitudes, identifying the deep/peak value as the qubit
   frequency.

So, mainly, the difference that this experiment introduces is a slightly more
complex pulse sequence. Therefore with start with that:

.. literalinclude:: ./includes/calibration/calibration1.py

.. image:: qubit_spectroscopy_light.svg
   :class: only-light
.. image:: qubit_spectroscopy_dark.svg
   :class: only-dark

Single shot classification
--------------------------

To avoid seeing other very similar experiment, let's jump to the single shot
classification experiment. The single-shot classification experiment is
conducted towards the end of the single-qubit calibration process and assumes
the availability of already calibrated pulses.

Two distinct pulse sequences are prepared for the experiment:

1. Sequence with only a measurement pulse.
2. Sequence comprising an RX pulse (X gate) followed by a measurement pulse.

For each sequence, the qubit is initialized in state 0 (no operation applied),
and a measurement is executed. This process is repeated multiple times. Unlike
previous experiments, the results of each individual measurement are saved
separately, avoiding averaging. Both measurements are repeated: first with the
single-pulse sequence and then with the two-pulse sequence. The goal is to
compare the outcomes and visualize the differences in the IQ plane between the
two states.

1. Prepare the single-pulse sequence: Measure the qubit multiple times in state
   0.
2. Prepare the two-pulse sequence: Apply an RX pulse followed by measurement,
   and perform the same measurement multiple times.
3. Plotting the Results: Plot the single-shot results for both sequences,
   highlighting the differences in the IQ plane between the two states.

This experiment serves to assess the effectiveness of single-qubit calibration
and its impact on qubit states in the IQ plane.

.. literalinclude:: ./includes/calibration/calibration2.py

.. image:: classification_light.svg
   :class: only-light
.. image:: classification_dark.svg
   :class: only-dark
