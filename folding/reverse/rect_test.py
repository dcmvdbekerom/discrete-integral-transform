import numpy as np
import matplotlib.pyplot as plt



x = np.arange(0,2,0.001)

for t in range(2,10):
    y = (1-x**t)**(1/t)
    plt.plot(x[1:],y[1:])
plt.show()
