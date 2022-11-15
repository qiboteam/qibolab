from qibolab import Platform

platform = Platform("rfsoc")

start = 4
npoints = 5
step = 1

results = []
for i in range(npoints):
    qd_pulse = platform.create_RX_pulse(0, start=0)
    ro_pulse = platform.create_MZ_pulse(0, start=qd_pulse.start)
    qd_pulse.duration = start + i * step

    sequence = PulseSequence()
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)

    avgi, avgq = platform.execute_pulse_sequence(sequence)
    results.append((avgi, avgq))

print(results)
