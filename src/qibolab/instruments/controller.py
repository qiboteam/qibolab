"""
Class to control general communication with instruments
"""
from qibolab.instruments.drivers import devices
from qibolab.instruments.drivers.abstract import DAC, LO, ADC

class InstrumentController:
    def __init__(self):
        self._instruments = []
        self._dac = []
        self._lo = []
        self._adc = []
        self._trigger_source = None

    def add_instrument(self, inst_type, name, address, trigger=False):
        """
        Searches for requested instrument and connects to it
        """
        inst = devices.get(inst_type)(name, address)
        self._instruments.append(inst)

        # There may be hybrid instruments with multiple capabilities, like the FPGA or the QRM
        if isinstance(inst, DAC):
            self._dac.append(inst)
        
        if isinstance(inst, ADC):
            self._adc.append(inst)

        if isinstance(inst, LO):
            self._lo.append(inst)

        # Experiment start trigger source
        if trigger:
            self._trigger_source = inst

    def fetch_instrument(self, name):
        """
        Fetches for instruemnt from added instruments
        """
        try:
            res = next(inst for inst in self._instruments if inst.name == name)
            return res
        except StopIteration:
            raise Exception("Instrument not found")

    def setup_instrument(self, name, parameters):
        """
        Setups instrument based on parameter dict
        """
        self.fetch_instrument(name).setup(**parameters)

    def translate_and_upload(self, name, sequence, shots):
        inst = self.fetch_instrument(name)
        payload = inst.translate(sequence, shots)
        inst.upload(payload)
        inst.play()

    def start_playback(self):
        for inst in self._lo:
            inst.start()

    def arm_adc(self, shots):
        for adc in self._adc:
            adc.arm(shots)

    def result(self, readout_if_dict):
        obj = {}
        for name, readout_if_frequencies in readout_if_dict.items():
            obj[name] = self.fetch_instrument(name).result(readout_if_frequencies)

        return obj

    def trigger_experiment(self):
        self._trigger_source.trigger()

    def stop(self):
        for inst in self._dac + self._lo:
            inst.stop()

    def close(self):
        for inst in self._instruments:
            inst.close()
