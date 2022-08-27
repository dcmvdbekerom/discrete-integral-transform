
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, log, exp, pi, sin, cos, sinh, cosh
from mpmath import jtheta
import sys
from scipy.special import wofz
log2 = log(2)

"""

The point of  this file is to have working Voigt: exact and Whiting approximation.

"""



gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gG  = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)

def gV_wofz(v,v0,wG,wL):
    gamma = wL/2
    sigma = wG/(2*sqrt(2*log2))
    x = v - v0
    z = (x + 1j*gamma)/(sigma*sqrt(2))
    V = wofz(z)/(sigma*sqrt(2*pi))
    return V.real

def gV_whiting(v, v0, wG, wL):
    d = (wL - wG) / (wL + wG)
    alpha = 0.18121
    beta = 0.023665*exp(0.6*d) + 0.00418*exp(-1.9*d)
    R = 1 - alpha*(1 - d**2) - beta*sin(pi*d)
    wV = R * (wL + wG)
    x = (v - v0) / wV
    wL_V = wL / wV

    Iw0 = 1/(wV*(1.065+0.447*wL_V + 0.058*wL_V**2))
    Iw1 = ((1 - wL_V) * np.exp(-2.772 * x**2) + wL_V * 1 / (1 + 4 * x**2)) * Iw0
    Iw2 = Iw1 + Iw0*0.016*(1-wL_V)*wL_V*(np.exp(-0.4*x**2.25)-10/(10+x**2.25))
    return Iw2

gG_FT = lambda x,w: np.exp(-(np.pi*x*w)**2/(4*np.log(2)))
gL_FT = lambda x,w: np.exp(- np.pi*np.abs(x)*w)
gV_FT = lambda x,wG,wL: gG_FT(x,wG)*gL_FT(x,wL)


dv = 0.001
Nv = 1000
v_arr = np.arange(-Nv,Nv)*dv
v_max = Nv*dv

x_arr = np.fft.rfftfreq(2*Nv, dv)
dx = x_arr[1]
x_max = Nv*dx



fig, ax = plt.subplots(1,2)


##for wG in [0.1, 0.2, 0.5, 1.0, 2.0]:    
##    for wL in [0.1, 0.2, 0.5, 1.0, 2.0]:
for wG in [1e-6, 1.0]:
    for wL in [1e-6, 1.0]:
        if (wL > 0.1) or (wG > 0.1):
            print(wL, wG)
            I0 = gV_wofz(v_arr,0.0,wG,wL)
            c = ax[0].plot(v_arr,I0,label='wG={:.1f}, wL={:.1f}'.format(wG,wL))[0].get_c()
            ax[0].plot(v_arr,gV_whiting(v_arr,0.0,wG,wL), 'k--')

ax[0].plot(v_arr[::100],gG(v_arr[::100],0.0,1.0),'+', label='G')
ax[0].plot(v_arr[::100],gL(v_arr[::100],0.0,1.0),'+', label='L')


##ax[0].plot(v_arr, gV_wofz(v_arr,0.0,1e-6,1.0))
##ax[0].plot(v_arr, gL(v_arr,0.0,1.0),'k--', label='L')
##
##ax[0].plot(v_arr, gV_wofz(v_arr,0.0,1.0,0.0))
##ax[0].plot(v_arr, gG(v_arr,0.0,1.0),'k--', label='G')

ax[0].legend()
plt.show()

