import numpy as np
import matplotlib.pyplot as plt

gauss = lambda t,w: np.exp(-0.5*(t/w)**2)


I = lambda x,y: gauss(x,2)*gauss(y,1)

t = np.linspace(-5,5,100)
xx,yy = np.meshgrid(t,t)

plt.contour(xx,yy,I(xx,yy))
plt.show()
