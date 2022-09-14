This PR implements the following changes:
- Remove calibration package and the calibration and diagnostics examples that are now transferred to qcvv.
- Clean up examples folder.
- Other improvements:
  - Add type hints and doc strings.
  - Declare class attributes in `__init__()`.
  - Rename private attributes to start with `_`.
- **Qutech `SPI` driver**
  - Rename file spi.py to qutech.py to be consistent with the naming convention used for instrument libraries.
  - Introduce parameter cache.
  - Simplify the runcard settings, removing unnecessary parameters.
  - Map device modules parameters under `dacs[channel]`, simplifying its access.
  - Replace `get_` & `set_` functions with properties  `dacs[channel].current` &  `dacs[channel].voltage`.
  - Provide an attribute `device` to allow access to all instrument features.
- **Rohde & Schwarz `SGS100A` driver**
  - Remove attributes not used by the instrument (for instance `repetition_duration`).
  - Remove `FrequencyParameter` required by `quantify`.
- `AbstractInstrument`
  - Remove `device` attribute, as this may not be needed by all instruments. The attribute was added to those instruments that need it.
- **Qblox drivers**
  - Temporarily remove support for Pulsars.
  - Implement new supporting classes `Sequencer` and `WaveformsBuffer` to simplify the code in `process_pulse_sequence()`.
  - `Cluster`
    - Introduce parameter cache.
    - Map attribute `reference_clock_source`.
  - `ClusterQRM_RF` and  `ClusterQCM_RF`
    - Add support for **multiplexed readout**.
    - Add support for **hardware modulation and demodulation**.
    - Simplify the structure of `acquisition_results` dictionary. In order to access there results, previously one needed to call `acquisition_results[sequencer][pulse.serial]`, now `acquisition_results[pulse.serial]`.
    - Implement an alternative way to obtain the **last** readout results of a qubit, using `acquisition_results[qubit]`.
    - Add `DEFAULT_SEQUENCERS` and `SAMPLING_RATE` constants.
    - Add `_port_channel_map` attribute from `channel_port_map`.
    - Add `channels` attribute.
    - Include additional parameters (`nshots`, `repetition_duration`, `hardware_mod_en` and `hardware_demod_en`) when generating a hash of the pulsesequence so that the sequencer's program is regenerated when they change.
    - Simplify the code in `process_pulse_sequence()` by making use of the new functions implemented in `PulseSequence`: `get_channel_pulses()`, `separate_overlapping_pulses()`.
    - Allocate and configure a sequencer for each output port. The rest of the sequencers are used as needed, either when the waveforms memory of the default sequencer runs out, or when having to play overlapping pulses.
    - Temporarily remove support for long pulses (those longer than 8192 ns).
    - Acquisition no longer starts immediately after the readout pulse (to later discard the first values), it now starts after the `acquisition_hold_off` period defined in the runcard.
 - **`Pulse`**
   - Add a new attribute `finish` that returns the point in time when the pulse finishes (start + duration).
   - Replace `phase` attribute with `relative_phase`, since taking care of the global sequence phase is now done by the `PulseShape`.
   - Remove the attributes `offset_i` and `offset_q`, as those belong to the instrument.
   - Attributes `start`, `duration` and `finish` accept symbolic expressions (more information below), allowing the use of these pulse attributes as variables in other pulses.
   - Add attributes `se_start`, `se_duration` and `se_finish` that return the those parameters as symbolic expressions.
   - Distinguish between `envelope_waveforms` and `modulated_waveforms`.
   - Add `copy()` and `shallow_copy()` methods.
   - Implement an enhanced `plot()` method:

![image](https://user-images.githubusercontent.com/55031026/190143593-73ec1866-0168-4dfd-91b1-7e8e368d0de0.png)

   - reinstated the `qubit` number to the pulse `serial`.
   - Add support for a small set of operators (`==`, `!=`, `+`, `*`). Pulse is hashable, but not unmutable (its hash depends on the current value of its parameters), so one can use the following operators to compare pulses:
```python
p0 != p1
p0 == p2
```
   - Adding two Pulses returns a `PulseSequence`. Multiplying a Pulse by an integer `n` returns a `PulseSequence` with `n` deep copies of the original pulse. For more examples, check the ![tutorial](https://github.com/qiboteam/qibolab/blob/alvaro/qblox_multiplex_pr/examples/tii/pulses_tutorial.ipynb).

 - **`PulseShape`**
   - Support the generation of waveforms using an arbitrary `SAMPLING_RATE`.
   - Fix a bug in the generation of `Drag` pulses.

 - **`Waveform`**
   - Implement a new supporting class to hold the data of pulse waveforms.
   - This class is hashable so is facilitates the identification of unique waveforms during the generation of the program.

 - **`PulseSequence`**
   - Transform `PulseSequence` class into a custom collection of `Pulse` objects, with many auxiliary methods.
   - Automatically sort the collection of pulses by `channel` and then by their `start` time.
   - The initialisation and the addition of pulses to a `PulseSequence` supports multiple pulses at once.
   - `PulseSequence` supports these operators `==`, `!=`, `+`, `+=`, `*`, `*=` with other `PulseSequence` objects and some also with `Pulse` objects.
   - Add `start`, `duration` and `finish` read only attributes
   - Add `pulses_overlap` attribute that returns True if the pulses of the sequence overlap.
   - Add `ro_pulses`, `qd_pulses` and `qf_pulses` attributes that return a new `PulseSequence` containing the subset of pulses of that type.
   - Add `channels` attribute returning a list of the channels required to play the sequence.
   - Implement the following support methods:
	- `add(*pulses)`
	- `append_at_end_of_channel(*pulses)`  appends each of the pulses at the end of their `channel`
	- `append_at_end_of_sequence(*pulses)` appends each of the pulses at the end of the `PulseSequence`
	- `index(pulse)` returns the index of a pulse
	- `pop(index=-1)` pops the last pulse in the `PulseSequence`
	- `remove(pulse)` removes pulse from the the `PulseSequence`
	- `clear()` removes all pulses in the `PulseSequence`
	- `shallow_copy()` makes a shallow copy of the `PulseSequence`, the new PulseSequence contains references to the same instances of the pulses as the original.
	- `deep_copy()` makes a deep copy of the `PulseSequence`, the new PulseSequence contains references to copies of the original pulses of the `PulseSequence`.
	- `get_channel_pulses(*channels)` returns a new `PulseSequence` containing the subset of pulses on those  `channels`.
	- `get_pulse_overlaps()` returns a dictionary of time intervals with the list of pulses in them
	- `separate_overlapping_pulses()` returns a list of `PulseSequence` objects in which the pulses don't overlap.
	- `plot()`:

![image](https://user-images.githubusercontent.com/55031026/190143417-263522ce-c90a-4bb8-aeac-0556b2860e51.png)

 For more examples on how to use the new features of `PulseSequence`, check the ![tutorial](https://github.com/qiboteam/qibolab/blob/alvaro/qblox_multiplex_pr/examples/tii/pulses_tutorial.ipynb).

- **`MultiqubitPlatform`**
  - Simplify the code with the new features provided by `PulseSequence`.

- **`AbstractPlatform`**
  - Implement a `transpilse()` method to convert native gates into pulses. This method supports virtual Z gates by keeping track of the phases accumulated on each qubit channel.
  - Rename all pulse generation functions to start with `create_`. For example `create_RX_pulse(...)`.
  - The pulses created using those pulse generation functions link the pulse to the qubit, by setting the `qubit` attribute of the pulse.
  - Move the initialisation of all attributes that do not require a connection to the instrument, to `__init__()`, so that these are available as soon as the platform (or backend) is initialised.
  - Add attribute `resonator_type` that can take values `2D` or `3D` (based on the number of qubits) that can later be used by the fitting functions (to determine whether to fit a maximum or a minimum).
  - Add attributes `qrm` (qubit readout module), `qcm` (qubit control module) and `qbm` (qubit biasing module) to provide access to the instruments responsible for controlling, reading or biasing each qubit:
```python
platform.qrm[qubit]
platform.qcm[qubit]
platform.qbm[qubit]
```
  - Add attributes `ro_port `, `qd_port ` and `qf_port ` that return a reference to the ports that control each qubit.
```python
platform.ro_port[qubit].attenuation = 20
platform.qd_port[qubit].lo_frequency = 100e6
```

- **Symbolic Expressions**
  - Add a new library that supports the use of *symbolic expressions* in `Pulse` and `PulseSequence` parameters. It is a much simpler and lightweight alternative to ![SymPy](https://www.sympy.org/en/index.html).

- **runcards**
  - Remove `minimum_delay_between_instructions` as this is a setting for a specific type of instruments (qblox) and is fixed.
  - Add a `roles` setting to each of the instruments, so that the platform can determine what actions to take during execution with each one of them. For example, for an instrument with `control` within its roles, the platform will call `play_sequence()`; on the other hand for a device with role `readout` it will call `play_sequence_and_acquire()`.
  - Replace characterisation parameters `resonator_spectroscopy_max_ro_voltage` and `rabi_oscillations_pi_pulse_min_voltage` with `state0_voltage` and `state1_voltage`.


- Fix a bug in transpilers `NativeGate`
Updated qibolab class diagram:
![qibolab](https://user-images.githubusercontent.com/55031026/190103750-1a7d162b-b5e5-4e1d-8549-32877a0d6d67.svg)

There is a detailed version of the diagram with all class attributes ![here](https://github.com/qiboteam/qibolab/tree/alvaro/qblox_multiplex_pr/doc/diagrams/exports).
