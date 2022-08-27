
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, log, exp, pi, sin, cos, sinh, cosh
from numpy.fft import rfft, irfft, rfftfreq, fftshift, ifftshift
from mpmath import jtheta
import sys
from scipy.special import wofz
log2 = log(2)


gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gG  = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)

def gV_wofz(v,v0,wG,wL):
    gamma = wL/2
    sigma = wG/(2*sqrt(2*log2))
    x = v - v0
    z = (x + 1j*gamma)/(sigma*sqrt(2))
    V = wofz(z)/(sigma*sqrt(2*pi))
    return V.real

def get_wV(wG, wL):
    d = (wL - wG) / (wL + wG)
    alpha = 0.18121
    beta = 0.023665*exp(0.6*d) + 0.00418*exp(-1.9*d)
    R = 1 - alpha*(1 - d**2) - beta*sin(pi*d)
    wV = R * (wL + wG)
    return wV

def gV_whiting(v, v0, wG, wL):
    wV = get_wV(wG,wL)
    x = np.abs(v - v0) / wV
    wL_V = wL / wV

    Iw0 = 1/(wV*(1.065+0.447*wL_V + 0.058*wL_V**2))
    Iw1 = ((1 - wL_V) * np.exp(-2.772 * x**2) + wL_V * 1 / (1 + 4 * x**2)) * Iw0
    Iw2 = Iw1 + Iw0*0.016*(1-wL_V)*wL_V*(np.exp(-0.4*x**2.25)-10/(10+x**2.25))
    return Iw2

gG_FT = lambda x,w: np.exp(-(np.pi*x*w)**2/(4*np.log(2)))
gL_FT = lambda x,w: np.exp(- np.pi*np.abs(x)*w)
gV_FT = lambda x,wG,wL: gG_FT(x,wG)*gL_FT(x,wL)


def gV_FT_cosh(x,wG,wL,I0_FT,err=1e-6):
    wV = get_wV(wG,wL)
    x_max = x[-1]
    
    n = 0
    I = gV_FT(x,wG,wL)
    while np.abs(I[-1] - I0_FT[-1]) > err:
        n += 1
        I += gV_FT(x - 2*n*x_max, wG, wL)
        I += gV_FT(x + 2*n*x_max, wG, wL)

    
    return I, n


def gV_FT_cos(x,wG,wL,I0_FT,err=1e-6):
    wV = get_wV(wG,wL)
    x_max = x[-1]
    dv = 1/(2*x_max)

    n = 0
    I = gV_wofz(0.0,0.0,wG,wL) * np.ones_like(x) * dv
    while np.abs(I[0] - I0_FT[0]) > err:
        # This works, but very slow because at x=0 there is always I'=/=0,
        # which cos(x) cannot resolve...
        n += 1
        I += 2 * gV_wofz(n*dv,0.0,wG,wL) * cos(2*pi*n*dv*(x_max-x)) * dv

    return I, n


def gV_FT_sin(x,wG,wL,I0_FT,err=1e-6):
    wV = get_wV(wG,wL)
    x_max = x[-1]
    dv = 1/(2*x_max)

    n = 0
    I = np.sum(gV_wofz(v_arr,0.0,wG,wL)) * np.zeros_like(x) * dv
    while np.abs(I[len(I)//2] - I0_FT[len(I)//2]) > err:
        n += 1
        I -= 2 * gV_wofz(n*dv,0.0,wG,wL) * sin(2*pi*n*dv*x) * dv

    return I, n


dv = 0.5
Nv = 10000
err = 1e-3
v_arr = np.arange(-Nv,Nv)*dv
v_max = Nv*dv

# Note we use 2x freq points to prevent circular convolution
x_arr = np.fft.rfftfreq(2*Nv, dv)
dx = x_arr[1]
x_max = Nv*dx



fig, ax = plt.subplots(1,2)

wV_list = []
n1_list = []
n2_list = []
wL = 0.0
for wL in [0.01,0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    for wG in [0.01,0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:    


        label = 'wG={:.1f}, wL={:.1f}'.format(wG,wL)
        wV = get_wV(wG,wL)
        wV_list.append(wV)
        
        I0 = gV_wofz(v_arr,0.0,wG,wL)
        I0[1::2]*=-1
        I0_FT = rfft(fftshift(I0*dv)).real
        
##        I1_FT,n1 = gV_FT_cosh(x_arr,wG,wL,I0_FT,err=err)
##        n1_list.append(n1)
        

##        I0[:Nv] *= -1
##        I0[Nv] = 0
##        I0_FT = rfft(fftshift(I0*dv)).imag
        
        I2_FT,n2 = gV_FT_cos(x_arr,wG,wL,I0_FT,err=err) 
        n2_list.append(n2)

        c = ax[0].plot(v_arr,I0,label=label)[0].get_c()
##        ax[0].plot(v_arr,gV_whiting(v_arr,0.0,wG,wL), 'k--')
        
        ax[1].plot(x_arr,I0_FT, label=label)
##        ax[1].plot(x_arr,I1_FT, 'k--', label=label)
        ax[1].plot(x_arr,I2_FT, 'k--', label=label)

##for i in np.argsort(wV_list):
##    print('wV = {:.2f} cm-1, n1 = {:2d}, n2 = {:2d}'.format(wV_list[i], n1_list[i], n2_list[i]))
##





####ax[0].legend()
##ax[1].set_yscale('log')
ax[1].set_ylim(1e-6,1e0)
plt.show()

