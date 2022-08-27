
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, log, exp, pi, sin, cos, sinh, cosh
from mpmath import jtheta
import sys

log2 = log(2)


gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gG  = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)

gL_FT = lambda x,x0,w: exp(-np.abs(x-x0)*pi*w)
gG_FT = lambda x,x0,w: exp(-(pi*(x-x0)*w)**2/(4*log2))
gV_FT = lambda x,x0,wG,wL: gG_FT(x,x0,wG)*gL_FT(x,x0,wL)

def gS(x,x_max,w, force_cosh=True, Nk=20):
    t = pi/log2 * (x_max*w)**2  # tau = 1j * t
    z = x/(2*x_max)
    if force_cosh: #t >= 1.0:
        print('cosh - ', end='')
        res = np.sum([gG_FT(x,2*k*x_max,w) for k in range(-Nk,Nk+1)],0)
    else:
        print('cos  - ', end='')
        dv = 1/(2*x_max)
        res = np.sum([gG(k*dv,0.0,w)*cos(2*pi*k*dv*x)*dv for k in range(-Nk,Nk+1)], 0)
    return res

dx = 0.001
N = 1000
x_max = N*dx
Nk_high = 20
Nk = 3
##w = 1.0

x_arr = np.arange(-x_max,x_max,dx)


plt.axhline(0,c='k')
plt.axvline(0,c='k')

for w in [0.2,0.5,1.0,2.0,5.0]:
    print('w:',w)
    I = gG_FT(x_arr,0.0,w)
    S0 = I + np.sum([gG_FT(x_arr,-2*k*x_max,w) + gG_FT(x_arr,2*k*x_max,w) for k in range(1,Nk_high)],0) 
    S1 = gS(x_arr,x_max,w, True, Nk=Nk)
    S2 = gS(x_arr,x_max,w, False, Nk=Nk)
##    c = plt.plot(v_arr, I, label=w)[0].get_c()
    plt.plot(x_arr, S0, 'k-', lw=3, alpha=0.5)#, c=c)
    c = plt.plot(x_arr, S1, '-')[0].get_c()
    plt.plot(x_arr, S2, '--', c=c)
    print('')
##plt.yscale('log')
plt.ylim(-0.05,0.65)
##plt.legend()
plt.show()


