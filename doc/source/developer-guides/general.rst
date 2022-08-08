Code overview
=============

The Qibolab framework implements a common system to deal with heterogeneous
platforms and custom experimental drivers.

Including a new platform
------------------------

New platforms can be implemented by inheriting the
:class:`qibolab.abstract.AbstractPlatform` and implementing its abstract
methods. In particular the developer should:

* Perform an ``AbstractPlatform`` inheritance.
* Create a yaml in ``src/qibolab/runcards/`` containing all the instruments and the calibration parameters
* Load your platform with:

.. code-block:: python

    from qibolab import Platform

    platform = Platform("<platform_name>", "<path_to_runcard>")

where ``<platform_name>`` is the name of the in the yaml file and
``<path_to_runcard>`` is the path to the yaml file.

Here you have an example for the structure of the yaml file:

.. code-block:: yaml

    name: icarusq
    nqubits: 1
    description: Controls the Tektronix AWG5204, AlazarTech ATS9371, ...

    settings:
        delay_before_readout: 0
        delay_between_pulses: 0
        resonator_spectroscopy_max_ro_voltage: 5726.62338856
        rabi_oscillations_pi_pulse_min_voltage: 5291.34802850
        experiment_start_instrument: awg

    instruments:
        awg:
            init_settings:
                name: awg
                address: TCPIP0::192.168.0.2::inst0::INSTR
            type: TektronixAWG5204
            settings:
                offset: [-0.001, 0, -0.002, 0.016]
                amplitude: [0.75, 0.75, 0.75, 0.75]
                resolution: 14
                sampling_rate: 2500000000
                mode: 1
                sequence_delay: 0.00006
                pulse_buffer: 0.000001
                adc_delay: 0.000000282
                qb_delay: 0.000000292
                ro_delay: 0.000000266
                ip: 192.168.0.2
                channel_phase_deg: [-6.2, 0.2, 10.6, -2.2]
                channel_phase: [-0.10821, 0.00349066, 0.1850049, -0.0383972]

        qb_lo:
            type: QuicSyn
            lo: true
            init_settings:
                name: qb_lo
                address: ASRL6::INSTR
            settings:
            frequency: 3866000000

        ro_lo:
            type: QuicSyn
            lo: true
            init_settings:
                name: ro_lo
            address: ASRL3::INSTR
            settings:
                frequency: 5083250000

        qb_att:
            type: MCAttenuator
            init_settings:
                name: qb_att
                address: 192.168.0.9:90
            settings:
                attenuation: 20

        ro_att:
            type: MCAttenuator
            init_settings:
                name: ro_att
                address: 192.168.0.10:100
            settings:
                attenuation: 15

        alazar_adc:
            type: AlazarADC
            adc: true
            init_settings:
                name: alazar_adc
                address: Alazar1
            settings:
                samples: 4992



When including a new platform, you should include its:

* **name:** The name of the new platform.
* **nqubits:** Maximum number of qubits supported by the platform.
* **description:** A brief description of the platform.
* **settings:** Platform settings.
* **instruments:** Dictionary with the instruments used by the platform with their setup parameters.
