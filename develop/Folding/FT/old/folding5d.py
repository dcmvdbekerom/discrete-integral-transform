
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, log, exp, pi
from mpmath import jtheta
import sys
log2 = log(2)

gG_FT = lambda x,x0,w: exp(-(pi*(x-x0)*w)**2/(4*log2))

def theta2(z,tau):
    return jtheta(3, z, exp(1j*pi*tau))

#TO-DO:
# 1. Apply to Voigt (as opposed to Gaussian)
# 2. Work back from theta function to cos/cosh


def gS(x,x_max,w):
    t = pi/log2 * (x_max*w)**2  # tau = 1j * t
    z = x/(2*x_max)
    if t >= 1.0:
        print('cosh - ', end='')
        z *= 1j * t
        A = gG_FT(x,0.0,w)
    else:
        print('cos  - ', end='')
        t = 1/t
        A = 1/(x_max*w)*sqrt(log2/pi)
        
    q = exp(-pi*t)
    res = A * jtheta(3,z*pi,q)
    return res


dx = 0.001
N = 1000
x_max = N*dx
xh = x_max/2
Nk = 20
##w = 1.0

x_arr = np.arange(-x_max,x_max,dx)


plt.axhline(0,c='k')
plt.axvline(0,c='k')

for w in [0.05, 0.1,0.2,0.5,1.0,2.0,5.0, 10.0, 20.0]:
    print('w:',w)
    I = gG_FT(x_arr,0.0,w)
    S = np.sum([gG_FT(x_arr,-2*k*x_max,w) + gG_FT(x_arr,2*k*x_max,w) for k in range(1,Nk)],0)
    S0 = float(gS(0.0,x_max,w).real)
    Sh = float(gS(xh,x_max,w).real)
    S1 = float(gS(x_max,x_max,w).real)

    c = plt.plot(x_arr, I, label=w)[0].get_c()
    plt.plot(x_arr, S+I, '--', c=c)
    plt.plot([-x_max,-xh,0.0,xh,x_max],[S1,Sh,S0,Sh,S1],'o',c=c, mfc=c, mec='k')
    print('')
##plt.yscale('log')
plt.ylim(-0.05,0.65)
plt.legend()
plt.show()


