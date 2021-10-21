from tiiq.instruments.abstract import Instrument


class Resonator():
    def __init__(self, frequency, ro_pulse_amplitude, ro_pulse_duration):
        self._frequency = frequency                         # Hz
        self._ro_pulse_amplitude = ro_pulse_amplitude       # %
        self._ro_pulse_duration = ro_pulse_duration         # s
    @property
    def frequency(self):
        return self._frequency
    @property
    def ro_pulse_amplitude(self):
        return self._ro_pulse_amplitude    
    @property
    def ro_pulse_duration(self):
        return self._ro_pulse_duration  
    
    def calibrate(self):
        pass

class Qubit():
    def __init__(self, frequency, pi_pulse_amplitude, pi_pulse_duration, T1, T2, resonator):
        self._frequency = frequency                         # Hz
        self._pi_pulse_amplitude = pi_pulse_amplitude       # %
        self._pi_pulse_duration = pi_pulse_duration         # s
        self._T1 = T1                                       # s
        self._T2 = T2                                       # s
        self._resonator = resonator
    @property
    def frequency(self):
        return self._frequency
    @property
    def pi_pulse_amplitude(self):
        return self._pi_pulse_amplitude  
    @property
    def pi_pulse_duration(self):
        return self._pi_pulse_duration    
    @property
    def T1(self):
        return self._T1
    @property
    def T2(self):
        return self._T2
    @property
    def resonator(self):
        return self._resonator

    def calibrate(self):
        pass


class Experiment():
    name = ""
    
    _resonators = []
    _qubits = []      
    _instruments = [] 

    _last_calibration_date = None

    # _diagram (image or url to schematic)

    def __init__(self, name, resonators, qubits, instruments):
        self.name = name
        experiments.append(self)
        self._resonators = resonators
        self._qubits = qubits
        self._instruments = instruments
        self._load_calibration()
        if self._last_calibration_date is None:
            self.calibrate()

    def calibrate(self):
        for resonator in self._resonators:
            resonator.calibrate()
        
        for qubit in self._qubits:
            qubit.calibrate()
        
        # self._last_calibration_date = now
        self._save_calibration()
        pass
    
    def _save_calibration():
        # calibration file path
        # write values to file
        pass

    def _load_calibration():
        # calibration file path
        # read file
        # assign values
        # update _last_calibrated
        pass

    @property
    def resonators(self):
        return self._resonators
    @property
    def qubits(self):
        return self._qubits
    @property
    def nqubits(self):
        return self._qubits.count
    @property
    def instruments(self):
        return self._instruments     



experiments = []
