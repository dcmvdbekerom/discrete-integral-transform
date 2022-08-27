
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, log, exp, pi
from mpmath import jtheta
import sys

gG = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)

def theta2(z,tau):
    return jtheta(3, z, exp(1j*pi*tau))

#TO-DO:
# 1. Apply to Voigt (as opposed to Gaussian)
# 2. Work back from theta function to cos/cosh


def gS(v,v_max,w):
    t = 4*log(2)/pi * (2*v_max/w)**2  # tau = 1j * t
    z = v/(2*v_max)
    if t >= 1.0:
        z *= 1j * t
        A = gG(v,0.0,w)
    else:
        t = 1/t
        A = 1/(2*v_max)
        
    q = exp(-pi*t)
    res = A * jtheta(3,z*pi,q)
    return res


dv = 0.001
N = 1000
v_max = N*dv
vh = v_max/2
Nk = 20
##w = 1.0

v_arr = np.arange(-v_max,v_max,dv)


plt.axhline(0,c='k')
plt.axvline(0,c='k')

for w in [0.2,0.5,1.0,2.0,5.0]:
    print('w:',w)
    I = gG(v_arr,0.0,w)
    S = np.sum([gG(v_arr,-2*k*v_max,w) + gG(v_arr,2*k*v_max,w) for k in range(1,Nk)],0)
    S0 = float(gS(0.0,v_max,w).real)
    Sh = float(gS(vh,v_max,w).real)
    S1 = float(gS(v_max,v_max,w).real)

    c = plt.plot(v_arr, I, label=w)[0].get_c()
    plt.plot(v_arr, S+I, '--', c=c)
    plt.plot([-v_max,-vh,0.0,vh,v_max],[S1,Sh,S0,Sh,S1],'o',c=c, mfc=c, mec='k')
    print('')
##plt.yscale('log')
plt.ylim(-0.05,0.65)
plt.legend()
plt.show()


