from qibolab.instruments.teledyne_scope import Teledyine_scope
import matplotlib.pyplot as plt


scope = Teledyine_scope("Qcomp_scope", "192.168.0.30")
scope.connect()

scope.set_trigger_level("C2", 1) # Volt
scope.set_hor_scale(1/(10e6)) # s/div
scope.set_ver_scale("C2", 0.5) #V/div

data = scope.get_timedwaveform("C2")

plt.figure()
plt.plot(data[0], data[1])
plt.show()