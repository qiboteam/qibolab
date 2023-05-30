from qibolab.channels import Channel
from qibolab.instruments.qblox.cluster import ClusterQCM, ClusterQCM_RF, ClusterQRM_RF
from qibolab.instruments.rohde_schwarz import SGS100A


class QbloxChannel(Channel):
    def __init__(self, name, instrument, port_name=None):
        self.name: str = name

        self.instrument = instrument
        if isinstance(instrument, (ClusterQRM_RF, ClusterQCM_RF, ClusterQCM)):
            if not port_name:
                raise ValueError(f"port_name argument is required for channels connected to qblox modules")

            self.port_name: str = port_name

    @property
    def lo_frequency(self):
        if isinstance(self.instrument, (ClusterQRM_RF, ClusterQCM_RF, ClusterQCM)):
            return self.instrument.ports[self.port_name].lo_frequency
        elif isinstance(self.instrument, SGS100A):
            return self.instrument.frequency
        else:
            raise ValueError(f"Instrument {type(self.instrument)} is not supported yet.")

    @lo_frequency.setter
    def lo_frequency(self, value):
        if isinstance(self.instrument, (ClusterQRM_RF, ClusterQCM_RF, ClusterQCM)):
            self.instrument.ports[self.port_name].lo_frequency = value
        elif isinstance(self.instrument, SGS100A):
            self.instrument.frequency = value
        else:
            raise ValueError(f"Instrument {type(self.instrument)} is not supported yet.")

    @property
    def lo_power(self):
        raise Exception("lo_power is not implemented in QbloxChannel")

    @lo_power.setter
    def lo_power(self, value):
        raise Exception("lo_power is not implemented in QbloxChannel")

    @property
    def attenuation(self):
        return self.instrument.ports[self.port_name].attenuation

    @attenuation.setter
    def attenuation(self, value):
        self.instrument.ports[self.port_name].attenuation = value

    @property
    def gain(self):
        return self.instrument.ports[self.port_name].gain

    @gain.setter
    def gain(self, value):
        self.instrument.ports[self.port_name].gain = value

    @property
    def offset(self):
        return self.instrument.ports[self.port_name].offset

    @offset.setter
    def offset(self, value):
        self.instrument.ports[self.port_name].offset = value

    # # proposed standard interfaces to access and modify instrument parameters

    # def set_lo_drive_frequency(self, qubit, freq):
    #     """Sets the frequency of the local oscillator used to upconvert drive pulses for a qubit."""
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     self.qd_port[qubit].lo_frequency = freq

    # def get_lo_drive_frequency(self, qubit):
    #     """Gets the frequency of the local oscillator used to upconvert drive pulses for a qubit."""
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     return self.qd_port[qubit].lo_frequency

    # def set_lo_readout_frequency(self, qubit, freq):
    #     """Sets the frequency of the local oscillator used to upconvert readout pulses for a qubit."""
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     self.ro_port[qubit].lo_frequency = freq

    # def get_lo_readout_frequency(self, qubit):
    #     """Gets the frequency of the local oscillator used to upconvert readout pulses for a qubit."""
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     return self.ro_port[qubit].lo_frequency

    # def set_attenuation(self, qubit, att):
    #     """Sets the attenuation of the readout port for a qubit."""
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     self.ro_port[qubit].attenuation = att

    # def get_attenuation(self, qubit):
    #     """Gets the attenuation of the readout port for a qubit."""
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     return self.ro_port[qubit].attenuation

    # def set_gain(self, qubit, gain):
    #     """Sets the gain of the drive port for a qubit."""
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     self.qd_port[qubit].gain = gain

    # def get_gain(self, qubit):
    #     """Gets the gain of the drive port for a qubit."""
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     return self.qd_port[qubit].gain

    # def set_bias(self, qubit, bias):
    #     """Sets the flux bias for a qubit.

    #     It supports biasing the qubit with a current source (SPI) or with the offset of a QCM module.
    #     """
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     if qubit in self.qbm:
    #         self.qb_port[qubit].current = bias
    #     elif qubit in self.qfm:
    #         self.qf_port[qubit].offset = bias

    # def get_bias(self, qubit):
    #     """Gets flux bias for a qubit."""
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     if qubit in self.qbm:
    #         return self.qb_port[qubit].current
    #     elif qubit in self.qfm:
    #         return self.qf_port[qubit].offset

    # # TODO: implement a dictionary of qubit - twpas
    # def set_lo_twpa_frequency(self, qubit, freq):
    #     """Sets the frequency of the local oscillator used to pump a qubit parametric amplifier."""
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     for instrument in self.instruments:
    #         if "twpa" in instrument:
    #             self.instruments[instrument].frequency = freq
    #             return None
    #     raise_error(NotImplementedError, "No twpa instrument found in the platform. ")

    # def get_lo_twpa_frequency(self, qubit):
    #     """Gets the frequency of the local oscillator used to pump a qubit parametric amplifier."""
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     for instrument in self.instruments:
    #         if "twpa" in instrument:
    #             return self.instruments[instrument].frequency
    #     raise_error(NotImplementedError, "No twpa instrument found in the platform. ")

    # def set_lo_twpa_power(self, qubit, power):
    #     """Sets the power of the local oscillator used to pump a qubit parametric amplifier."""
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     for instrument in self.instruments:
    #         if "twpa" in instrument:
    #             self.instruments[instrument].power = power
    #             return None
    #     raise_error(NotImplementedError, "No twpa instrument found in the platform. ")

    # def get_lo_twpa_power(self, qubit):
    #     """Gets the power of the local oscillator used to pump a qubit parametric amplifier."""
    #     qubit = qubit.name if isinstance(qubit, Qubit) else qubit
    #     for instrument in self.instruments:
    #         if "twpa" in instrument:
    #             return self.instruments[instrument].power
    #     raise_error(NotImplementedError, "No twpa instrument found in the platform. ")
