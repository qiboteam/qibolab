from qblox_instruments.qcodes_drivers.qcm_qrm import QcmQrm as QbloxQrmQcm
from qibo.config import log, raise_error

from qibolab.instruments.port import Port

FREQUENCY_LIMIT = 500e6


class QbloxOutputPort(Port):
    def __init__(self, sequencer_number: int, number: int):
        self.device: QbloxQrmQcm = None
        self.module = None
        self.sequencer_number: int = sequencer_number
        self.port_number: int = number - 1
        self.channel = None  # To be discontinued
        self.qubit = None  # To be discontinued

    # qibolab.instruments.port.Port interface implementation

    @property
    def attenuation(self) -> str:
        """Attenuation that is applied to this port."""

        return self.device.get(f"out{self.port_number}_att")

    @attenuation.setter
    def attenuation(self, value):
        if isinstance(value, float):
            value = int(value)
        if isinstance(value, int):
            if value > 60:
                log.warning(f"Qblox attenuation needs to be between 0 and 60 dB. Adjusting {value} to 60dB")
                value = 60

            elif value < 0:
                log.warning(f"Qblox attenuation needs to be between 0 and 60 dB. Adjusting {value} to 0")
                value = 0

            if (value % 2) != 0:
                log.warning(
                    f"Qblox attenuation needs to be a multiple of 2 dB. Adjusting {value} to {round(value/2) * 2}"
                )
                value = round(value / 2) * 2
        else:
            raise_error(ValueError, f"Invalid attenuation {value}")

        self.module._set_device_parameter(self.device, f"out{self.port_number}_att", value=value)

    @property
    def offset(self):
        """DC offset that is applied to this port."""

        return self.device.get(f"out{self.port_number}_offset")

    @offset.setter
    def offset(self, value):
        if isinstance(value, int):
            value = float(value)
        if isinstance(value, float):
            if value > 2.5:
                log.warning(f"Qblox offset needs to be between -2.5 and 2.5 V. Adjusting {value} to 2.5 V")
                value = 2.5

            elif value < -2.5:
                log.warning(f"Qblox offset needs to be between -2.5 and 2.5 V. Adjusting {value} to -2.5 V")
                value = -2.5
        else:
            raise_error(ValueError, f"Invalid offset {value}")

        self.module._set_device_parameter(self.device, f"out{self.port_number}_offset", value=value)

    # Additional attributes needed by the driver
    @property
    def hardware_mod_en(self):
        """Flag to enable hardware modulation."""

        return self.device.sequencers[self.sequencer_number].get(f"mod_en_awg")

    @hardware_mod_en.setter
    def hardware_mod_en(self, value):
        if not isinstance(value, bool):
            raise_error(ValueError, f"Invalid hardware_mod_en {value}")

        self.module._set_device_parameter(self.device.sequencers[self.sequencer_number], "mod_en_awg", value=value)

    @property
    def nco_freq(self):
        """nco_freq that is applied to this port."""

        return self.device.sequencers[self.sequencer_number].get(f"nco_freq")

    @nco_freq.setter
    def nco_freq(self, value):
        if isinstance(value, float):
            value = int(value)
        if isinstance(value, int):
            if value > FREQUENCY_LIMIT:
                log.warning(
                    f"Qblox nco_freq needs to be between -{FREQUENCY_LIMIT} and {FREQUENCY_LIMIT} MHz. Adjusting {value} to {FREQUENCY_LIMIT} MHz"
                )
                value = int(FREQUENCY_LIMIT)

            elif value < -FREQUENCY_LIMIT:
                log.warning(
                    f"Qblox nco_freq needs to be between -{FREQUENCY_LIMIT} and {FREQUENCY_LIMIT} MHz. Adjusting {value} to -{FREQUENCY_LIMIT} MHz"
                )
                value = int(-FREQUENCY_LIMIT)
        else:
            raise_error(ValueError, f"Invalid nco_freq {value}")

        self.module._set_device_parameter(self.device.sequencers[self.sequencer_number], "nco_freq", value=value)

    @property
    def nco_phase_offs(self):
        """nco_phase_offs that is applied to this port."""

        return self.device.sequencers[self.sequencer_number].get(f"nco_phase_offs")

    @nco_phase_offs.setter
    def nco_phase_offs(self, value):
        if isinstance(value, int):
            value = float(value)
        if isinstance(value, float):
            value = value % 360
        else:
            raise_error(ValueError, f"Invalid nco_phase_offs {value}")

        self.module._set_device_parameter(self.device.sequencers[self.sequencer_number], "nco_phase_offs", value=value)


class ClusterRF_OutputPort(QbloxOutputPort):
    @property
    def lo_enabled(self):
        """Flag to enable local oscillator."""

        if self.device.is_qrm_type:
            return self.device.get(f"out{self.port_number}_in{self.port_number}_lo_en")
        elif self.device.is_qcm_type:
            return self.device.get(f"out{self.port_number}_lo_en")

    @lo_enabled.setter
    def lo_enabled(self, value):
        if not isinstance(value, bool):
            raise_error(ValueError, f"Invalid lo_enabled {value}")

        if self.device.is_qrm_type:
            self.module._set_device_parameter(self.device, f"out{self.port_number}_in{self.port_number}_lo_en", value=value)
        elif self.device.is_qcm_type:
            self.module._set_device_parameter(self.device, f"out{self.port_number}_lo_en", value=value)


    @property
    def lo_frequency(self):
        """Local oscillator frequency for the given port."""
        if self.device.is_qrm_type:
            return self.device.get(f"out{self.port_number}_in{self.port_number}_lo_freq")
        elif self.device.is_qcm_type:
            return self.device.get(f"out{self.port_number}_lo_freq")

    @lo_frequency.setter
    def lo_frequency(self, value):
        if isinstance(value, float):
            value = int(value)
        if isinstance(value, int):
            if value > 18e9:
                log.warning(f"Qblox lo_frequency needs to be between 2e9 and 18e9 Hz. Adjusting {value} to 18e9 Hz")
                value = int(18e9)

            elif value < 2e9:
                log.warning(f"Qblox lo_frequency needs to be between 2e9 and 18e9 Hz. Adjusting {value} to 2e9 Hz")
                value = int(2e9)
        else:
            raise_error(ValueError, f"Invalid lo-frequency {value}")
        if self.device.is_qrm_type:
            self.module._set_device_parameter(self.device, f"out{self.port_number}_in{self.port_number}_lo_freq", value=value)
        elif self.device.is_qcm_type:
            self.module._set_device_parameter(self.device, f"out{self.port_number}_lo_freq", value=value)

    # Note: for qublos, gain is equivalent to amplitude, since it does not bring any advantages
    # we plan to remove it soon.
    @property
    def gain(self):
        """Gain that is applied to this port."""

        return self.device.sequencers[self.sequencer_number].get(f"gain_awg_path0")

    @gain.setter
    def gain(self, value):
        if isinstance(value, int):
            value = float(value)
        if isinstance(value, float):
            if value > 1.0:
                log.warning(f"Qblox offset needs to be between -1 and 1. Adjusting {value} to 1")
                value = 1.0

            elif value < -1.0:
                log.warning(f"Qblox offset needs to be between -1 and 1. Adjusting {value} to -1")
                value = -1.0
        else:
            raise_error(ValueError, f"Invalid offset {value}")

        self.module._set_device_parameter(
            self.device.sequencers[self.sequencer_number], "gain_awg_path0", "gain_awg_path1", value=value
        )


class ClusterBB_OutputPort(QbloxOutputPort):
    # Note: for qublos, gain is equivalent to amplitude, since it does not bring any advantages
    # we plan to remove it soon.
    @property
    def gain(self):
        """Gain that is applied to this port."""

        return self.device.sequencers[self.sequencer_number].get(f"gain_awg_path0")

    @gain.setter
    def gain(self, value):
        if isinstance(value, int):
            value = float(value)
        if isinstance(value, float):
            if value > 1.0:
                log.warning(f"Qblox offset needs to be between -1 and 1. Adjusting {value} to 1")
                value = 1.0

            elif value < -1.0:
                log.warning(f"Qblox offset needs to be between -1 and 1. Adjusting {value} to -1")
                value = -1.0
        else:
            raise_error(ValueError, f"Invalid offset {value}")

        self.module._set_device_parameter(self.device.sequencers[self.sequencer_number], "gain_awg_path0", value=value)


class QbloxInputPort(QbloxPort):
    def __init__(self, output_sequencer_number: int, input_sequencer_number: int, number: int):
        self.device: QbloxQrmQcm = None
        self.module = None
        self.output_sequencer_number: int = output_sequencer_number
        self.input_sequencer_number: int = input_sequencer_number
        self.port_number: int = number - 1
        self.channel = None  # To be discontinued
        self.qubit = None  # To be discontinued

        self.acquisition_hold_off = 4  # To be discontinued

    @property
    def hardware_demod_en(self):
        """Flag to enable hardware demodulation."""

        return self.device.sequencers[self.input_sequencer_number].get(f"demod_en_acq")

    @hardware_demod_en.setter
    def hardware_demod_en(self, value):
        if not isinstance(value, bool):
            raise_error(ValueError, f"Invalid hardware_demod_en {value}")

        self.module._set_device_parameter(self.device.sequencers[self.input_sequencer_number], "demod_en_acq", value=value)

    @property
    def acquisition_duration(self):
        """Duration of the pulse acquisition, in ns. It must be > 0 and multiple of 4."""

        return self.device.sequencers[self.output_sequencer_number].get(f"integration_length_acq")

    @acquisition_duration.setter
    def acquisition_duration(self, value):
        if isinstance(value, float):
            value = int(value)
        if isinstance(value, int):
            if value < 4:
                log.warning(f"Qblox hardware_demod_en needs to be > 4ns. Adjusting {value} to 4 ns")
                value = 4
            if (value % 4) != 0:
                log.warning(
                    f"Qblox hardware_demod_en needs to be a multiple of 4 ns. Adjusting {value} to {round(value/4) * 4}"
                )
                value = round(value / 4) * 4

        else:
            raise_error(ValueError, f"Invalid acquisition_duration {value}")

        self.module._set_device_parameter(
            self.device.sequencers[self.output_sequencer_number], "integration_length_acq", value=value
        )
