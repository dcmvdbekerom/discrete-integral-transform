import numpy as np
import matplotlib.pyplot as plt

gG0 = lambda x: np.exp(-(np.pi*x)**2/(4*np.log(2)))
gG1 = lambda x: -(np.pi*x)**2/(2*np.log(2)) * gG0(x)
gG2 = lambda x: (np.pi*x)**2/(2*np.log(2))*((np.pi*x)**2/(2*np.log(2))-2) * gG0(x)

gL0 = lambda x: np.exp(-np.abs(np.pi*x))

dx = 0.01
x_max = 10.0
x = np.arange(-x_max/2,x_max/2,dx)

ksi = np.log(x)

gG1a = np.ediff1d(gG0(x),to_end = 0)/dx
gG2a = np.ediff1d(gG1a,to_end = 0)/dx


plt.plot(x,gG0(x))
plt.plot(x,gG1a)
plt.plot(x,gG1(x),'--')
plt.plot(x,gG2a)
plt.plot(x,gG2(x),'--')
plt.show()






##
##plt.plot(x,gL_FT(x))
##plt.show()
