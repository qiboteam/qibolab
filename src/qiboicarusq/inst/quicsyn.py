from qiboicarusq.inst import Instrument

class QuicSyn(Instrument):

    def setup(self):
        self.write('0601') # EXT REF

    def set_frequency(self, frequency):
        """
        Sets the frequency in Hz
        """
        self.write('FREQ {0:f}Hz'.format(frequency))

    def start(self):
        self.write('0F01')

    def stop(self):
        self.write('0F00')
