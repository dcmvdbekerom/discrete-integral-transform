import numpy as np
import matplotlib.pyplot as plt

sinc = lambda x: np.sin(np.pi*x)/(np.pi*x)

x = np.arange(0.0,100,0.5)
y = sinc(x)

plt.axhline(0,c='k')
plt.plot(x,y,'.-')
plt.show()
