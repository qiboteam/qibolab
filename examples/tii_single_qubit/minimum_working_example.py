from qibolab import Platform
from qibolab.pulses import Pulse, ReadoutPulse
from qibolab.circuit import PulseSequence
from qibolab.pulse_shapes import Rectangular, Gaussian

# Define PulseSequence
sequence = PulseSequence()
# Add some pulses to the pulse sequence
sequence.add(Pulse(start=0,
                   frequency=200_000_000,
                   amplitude=0.3,
                   duration=4000,
                   phase=0,
                   shape=Gaussian(5), # Gaussian shape with std = duration / 5
                   channel=1)) 

sequence.add(ReadoutPulse(start=4004,
                          frequency=20_000_000,
                          amplitude=0.9,
                          duration=2000,
                          phase=0,
                          shape=Rectangular(), 
                          channel=11)) 

# Define platform and load specific runcard
platform = Platform("tiiq")
# Connects to lab instruments using the details specified in the calibration settings.
platform.connect()
# Configures instruments using the loaded calibration settings.
platform.setup()
# Turns on the local oscillators
platform.start()
# Executes a pulse sequence.
results = platform.execute(sequence, nshots=3000)
print(f"results (amplitude, phase, i, q): {results}")
# Turn off lab instruments
platform.stop()
# Disconnect from the instruments
platform.disconnect()