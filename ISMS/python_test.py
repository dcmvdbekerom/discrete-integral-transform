from radis.lbl.broadening import olivero_1977, voigt_lineshape
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10*np.pi, 100)
y = np.sin(x)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'b-')
phase_arr = np.linspace(0, 10*np.pi, 100)
amp_arr = np.linspace(1,1.5,100)
for phase,amp in zip(phase_arr,amp_arr):
	line1.set_ydata(amp*np.sin(0.5 * x + phase))
	fig.canvas.draw()
	fig.canvas.flush_events()
