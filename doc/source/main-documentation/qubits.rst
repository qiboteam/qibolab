Qubits
======

The :class:`qibolab.qubits.Qubit` is a representation of a physical qubit, it mainly contains three elements:

    - channels
    - parameters
    - native_gates

The channels, better explained in :doc:`/main-documentation/channels`, represent the physical wires in a laboratory.
The channels are all optional and come in different types:

    - readout (from controller device to the qubits)
    - feedback (from qubits to controller)
    - twpa (pump to the twpa)
    - drive
    - flux
    - flux_coupler

The settable parameters, that are read from the runcard when the platform is initialized, are:

    - bare_resonator_frequency
    - readout_frequency
    - drive_frequency
    - anharmonicity
    - Ec
    - Ej
    - g
    - assigment_fidelity
    - sweetspot
    - peak_vol`tage
    - pi_pulse_amplitude
    - T1
    - T2
    - T2_spin_echo
    - state0_voltage
    - state1_voltage
    - mean_gnd_states
    - mean_exc_states
    - threshold
    - iq_angle
