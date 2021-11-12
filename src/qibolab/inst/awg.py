import time
from typing import Union, Optional, Tuple, List
import broadbean as bb
import numpy as np
from qcodes.instrument_drivers.tektronix.AWG70000A import AWG70000A
from qibolab.inst import Instrument

class AWG5204(Instrument):
    def __init__(self, address:str='TCPIP0::192.168.0.2::inst0::INSTR', timeout:int = 240000, sampling_rate=2.4e9):
        super().__init__(address, timeout)
        self._nchannels = 4
        self._sampling_rate = sampling_rate
        self._amplitude = [0.75, 0.75, 0.75, 0.75]
    
    def setup(self,
              offset: List[Union[int, float]],
              amplitude: Optional[List[Union[int, float]]] = [0.75, 0.75, 0.75, 0.75],
              resolution: Optional[int] = 14,
              sampling_rate: Optional[Union[int, float]] = 2.4e9) -> None:
        """ Sets the channel offset, maximum amplitude, DAC resolution and sampling rate of the AWG
        """

        self.reset()
        for idx in range(self._nchannels):
            ch = idx + 1
            self.write("SOURCe{}:VOLTage {}".format(ch, amplitude[idx]))
            self._amplitude[idx] = amplitude[idx]
            self.write("SOURCE{}:VOLTAGE:LEVEL:IMMEDIATE:OFFSET {}".format(ch, offset[ch - 1]))
            self.write("SOURce{}:DAC:RESolution {}".format(ch, resolution))

        self.write("SOUR3:DMOD MIX")
        self.write("SOUR4:DMOD MIX")
        self.write("CLOCk:SRATe {}".format(sampling_rate))
        self._sampling_rate = sampling_rate
        self.query("*OPC?")

    def reset(self) -> None:
        self.write("INSTrument:MODE AWG")
        self.write("CLOC:SOUR EFIX") # Set AWG to external reference, 10 MHz
        self.write("CLOC:OUTP:STAT OFF") # Disable clock output
        self.clear()

    def clear(self) -> None:
        self.write('SLISt:SEQuence:DELete ALL')
        self.write('WLISt:WAVeform:DELete ALL')
        self.wait_ready()

    def wait_ready(self) -> None:
        self.query("*OPC?")

    def upload_sequence(self, sequence: np.ndarray, repetitions: int) -> None:
        """ Uploads a sequence of waveforms to the AWG
        
        Args:
            sequence: np.ndarray of [step, channel, wfm/m1/m2, amplitude],
            repetitions: number of averaging
        """
        main_sequence = bb.Sequence()
        main_sequence.name = "MainSeq"
        main_sequence.setSR(self._sampling_rate)
        
        steps = len(sequence)
        for step in range(steps):
            seq = sequence[step]
            subseq = bb.Sequence()
            subseq.setSR(self._sampling_rate)

            subseq_wfm = bb.Element()
            for idx in range(self._nchannels):
                ch = idx + 1
                wfm = seq[idx, 0]
                m1 = seq[idx, 1]
                m2 = seq[idx, 2]
                subseq_wfm.addArray(ch, wfm, self._sampling_rate, m1=m1, m2=m2)

            subseq.addElement(1, subseq_wfm)
            main_sequence.addSubSequence(step + 1, subseq)
            main_sequence.setSequencingTriggerWait(step + 1, 1)
            main_sequence.setSequencingNumberOfRepetitions(step + 1, repetitions)

        main_sequence.setSequencingGoto(steps, 1) # Set loopback
        payload = main_sequence.forge(apply_delays=False, apply_filters=False)
        payload = AWG70000A.make_SEQX_from_forged_sequence(payload, self._amplitude, "MainSeq")

        with open("//192.168.0.2/Users/OEM/Documents/MainSeq.seqx", "wb+") as w:
            w.write(payload)

        pathstr = 'C:\\Users\\OEM\\Documents\\MainSeq.seqx'
        self.write('MMEMory:OPEN:SASSet:SEQuence "{}"'.format(pathstr))

        start = time.time()
        while True:
            elapsed = time.time() - start
            if int(self.query("*OPC?")) == 1:
                break
            elif elapsed > self._visa_handle.timeout:
                raise RuntimeError("AWG took too long to load waveforms")

        for ch in range(self._nchannels):
            self.write('SOURCE{}:CASSet:SEQuence "MainSeq", {}'.format(ch + 1, ch + 1))
        self.wait_ready()
        self.ready()
        self.wait_ready()

    def ready(self):
        for ch in range(self._nchannels):
            self.write("OUTPut{}:STATe 1".format(ch + 1))
            self.write('SOURce{}:RMODe TRIGgered'.format(ch + 1))
            self.write('SOURce1{}TINPut ATRIGGER'.format(ch + 1))

        # Arm the trigger
        self.write('AWGControl:RUN:IMMediate')

    def stop(self):
        self.write('AWGControl:STOP')
        for ch in range(self._nchannels):
            self.write("OUTPut{}:STATe 0".format(ch + 1))

    def trigger(self):
        self.write('TRIGger:IMMediate ATRigger')
