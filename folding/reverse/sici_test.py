import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sici

dx = 0.01
x_max = 10
x = np.arange(-x_max,x_max,dx)

for a in np.arange(1,10,1):
    Si, Ci = sici(x)
    Sim, Cim = sici(x-a*1j)
    Sip, Cip = sici(x+a*1j)
    Sia, Cia = sici(a*1j)

    plt.plot(x, (Sim + Sip).real)
##plt.plot(x, np.sin(x/2)/x)
plt.grid()
plt.show()

