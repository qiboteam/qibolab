# -*- coding: utf-8 -*-
from qibolab import Platform
from qibolab.paths import qibolab_folder

# Define platform and load specific runcard
runcard = qibolab_folder / "runcards" / "tii5q.yml"
platform = Platform("tii5q", runcard)

# Connects to lab instruments using the details specified in the calibration settings.
platform.connect()
# Configures instruments using the loaded calibration settings.
platform.setup()
# Turns on the local oscillators
platform.start()
# Executes a pulse sequence.
results = platform.measure_fidelity(nshots=3000)
print(f"results[qubit] (rotation_angle, threshold, fidelity, assignment_fidelity): {results}")
# Turn off lab instruments
platform.stop()
# Disconnect from the instruments
platform.disconnect()