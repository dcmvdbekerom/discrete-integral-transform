
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
        A = gG_FT(x,0.0,w)
##        res = A*(1 + 2*np.sum([exp(-(pi*2*k*x_max*w)**2/(4*log2))*cosh(4*k*x*x_max*pi**2*w**2/(4*log2)) for k in range(1,20)], 0))
        res = np.sum([gG_FT(x,2*k*x_max,w) for k in range(-Nk,Nk+1)],0)
    else:
        print('cos  - ', end='')
        A = 1/(x_max*w)*sqrt(log2/pi)
##        res = A*(1 + 2*np.sum([exp(-4*log2*(k/(2*x_max*w))**2)*cos(pi*k*x/x_max) for k in range(1,20)], 0))
        res = A*np.sum([exp(-4*log2*(k/(2*x_max*w))**2)*cos(pi*k*x/x_max) for k in range(-Nk,Nk+1)], 0)

        gG  = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)




##        res = A*(1 + 2*np.sum([exp(-4*log2*(k*dv/w)**2)*cos(2*pi*k*dv*x) for k in range(1,20)], 0))
##        res = A*(1 + 2*np.sum([exp(-4*log2*(v/w)**2)*cos(2*pi*v*x) for k in range(1,20)], 0))
    return res

dx = 0.001
N = 1000
x_max = N*dx
Nk = 20
##w = 1.0

x_arr = np.arange(-x_max,x_max,dx)


plt.axhline(0,c='k')
plt.axvline(0,c='k')

for w in [0.2,0.5,1.0,2.0,5.0]:
    print('w:',w)
    I = gG_FT(x_arr,0.0,w)
    S0 = I + np.sum([gG_FT(x_arr,-2*k*x_max,w) + gG_FT(x_arr,2*k*x_max,w) for k in range(1,Nk)],0) 
    S1 = gS(x_arr,x_max,w, True)
    S2 = gS(x_arr,x_max,w, False)
##    c = plt.plot(v_arr, I, label=w)[0].get_c()
    plt.plot(x_arr, S1, '-')#, c=c)
    plt.plot(x_arr, S2, 'k--')
    print('')
##plt.yscale('log')
plt.ylim(-0.05,0.65)
##plt.legend()
plt.show()


