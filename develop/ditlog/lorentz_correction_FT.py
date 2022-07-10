import numpy as np
import matplotlib.pyplot as plt
from ditlog import synthesize_spectrum as spectrum
from numpy.fft import rfftfreq, rfft, irfft, fftshift, ifftshift
from numpy import log,exp,pi,sqrt
from math import factorial
from scipy.optimize import curve_fit
import sys

gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gG  = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)
def gV(v,v0,wG,wL):
    gamma = (wG**5 + 2.69269*wG**4*wL + 2.42843*wG**3*wL**2 + 4.47163*wG**2*wL**3 + 0.07842*wG*wL**4 + wL**5)**0.2
    eta   = 1.36603*wL/gamma - 0.47719*(wL/gamma)**2 + 0.11116*(wG/gamma)**3
    return (1-eta) * gG(v,v0,gamma) + eta * gL(v,v0,gamma)

gL_FT  = lambda x,w: np.exp(-np.abs(x)*pi*w)
gG_FT  = lambda x,w: np.exp(-(x*pi*w)**2/(4*log(2)))
gE = lambda v,v0,w: np.exp(-np.abs(v-v0)/w)
corr_factor = lambda x,B,A,tau: A*np.exp(x/tau) + B
gE_FT = lambda x,w: 2*w / (1 + 4*pi**2*x**2*w**2) 


def gL_FT_corr(x,w):
    print(x)
    res = gL_FT(x,w)/dv

    log_wvmax = np.log(w/(Nv*dv))
    poptA = [ 6.74408847, 0.23299924, 0.53549119]
    poptB = [ 6.82193751, 0.09226203, 0.49589094]
    poptw = [-0.93309186, 0.18724358, 0.50806536]

    A = w*np.exp(-corr_factor(log_wvmax,*poptA))
    B = w*np.exp(-corr_factor(log_wvmax,*poptB))
    w_corr = (Nv*dv)*np.exp(corr_factor(log_wvmax,*poptw))

    Err_corr = A*gE_FT(x, w_corr/dv)/(Nv*dv)**2
    Err_corr[0] += B*200/(Nv*dv)/dv
    I_alternate = 1-(np.arange(len(x))&1)*2
    Err_corr *= I_alternate

    res -= Err_corr
    return res

gV_FT = lambda x,wG,wL: gL_FT(x,wL)*gG_FT(x,wG)
gV_FT_corr = lambda x,wG,wL: gL_FT_corr(x,wL)*gG_FT(x,wG)

 
def gV_FT_folding(x, wG, wL, g_FT, folding_thresh=1e-6):

    x_fold = (x, x[::-1])
    result = np.zeros(x.size)
    n = 0
    while gV_FT(n/2,wG,wL) >= folding_thresh:
        result += g_FT(n/2 + x_fold[n&1], wG, wL)
        n += 1
        
    return result

vmin = 1800
vmax = 2200
dv = 0.01
wG = 2.0
wL = 5.0

v_arr = np.arange(vmin,vmax,dv)
Nv = len(v_arr)
x_arr = rfftfreq(2*Nv,1)

I_direct = gV(v_arr,vmin,wG,wL)
I_DFT = gV_FT_folding(x_arr,wG/dv,wL/dv,gV_FT)/dv
I_IDFT = irfft(I_DFT)[:Nv]

I_DFT_corr = gV_FT_folding(x_arr,wG/dv,wL/dv,gV_FT_corr)/dv
I_IDFT_corr = irfft(I_DFT_corr)[:Nv]


###Compare FT's:
plt.plot(x_arr[:Nv//2+1],rfft(fftshift(I_IDFT-I_direct)).real,'k',label='DFT',lw=3)
plt.plot(x_arr,Err_corr * I_alternate,'r',label='Exact (corrected)')
plt.yscale('log')
plt.legend()
plt.show()

###Compare plots:
##plt.plot(v_arr - vmin, I_direct,'k',label='Direct',lw=3)
##plt.plot(v_arr - vmin, I_IDFT,'r--',label='IDFT naive')
##plt.plot(v_arr - vmin, I_IDFT_corr,'r',label='IDFT corrected')
##plt.yscale('log')
##plt.legend()
##plt.show()

