import numpy as np
import matplotlib.pyplot as plt

g = lambda x,w: np.exp(-w*x)


x_max = 10
dx = 0.01

x = np.arange(dx,x_max,dx)

dxE = 1.2

p  = 1.2

wp2 = p**2
wp1 = p**1
we =  p**0
wm1 = p**(-1)
wm2 = p**(-2)

Ip2 = g(x,wp2)
Ip1 = g(x,wp1)
Ie  = g(x,we)
Im1 = g(x,wm1)
Im2 = g(x,wm2)

Ia1 = 0.5*Ip1 + 0.5*Im1
Ia2 = 0.5*Ip2 + 0.5*Im2


Ia3 = (-0.5*Ip2 + 4*0.5*Ip1 + 4*0.5*Im1 - 0.5*Im2)/3


##plt.plot(x,Ip1)
##plt.plot(x,Im1)

plt.plot(x,Ie)
plt.plot(x,Ia1,'--',label = '1')
##plt.plot(x,Ia2,'--',label='2')
plt.plot(x,Ia3,'--',label='3')

plt.legend()

##plt.plot(x,Ia1-Ie,'-')
##plt.plot(x,Ia2-Ie,'-')
##plt.plot(x,Ia3-Ie,'--')
plt.axhline(0.0002)
plt.show()
