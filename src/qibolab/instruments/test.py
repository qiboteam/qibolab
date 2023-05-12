from erasynth import ERA

era = ERA("era", "192.168.0.207")

era.connect()
print(era.frequency)
print(era.power)
era.frequency = 1e9
era.power = -2.3
print(era.frequency)
print(era.power)

era.start()
from time import sleep
print("start")
# sleep(10)
era.stop()
era.disconnect()
