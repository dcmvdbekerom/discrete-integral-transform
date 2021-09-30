import numpy as np
import matplotlib.pyplot as plt
from ditlog import synthesize_spectrum as spectrum
from numpy.fft import rfftfreq, rfft, irfft, fftshift, ifftshift
from numpy import log,exp,pi,sqrt
from math import factorial
from scipy.optimize import curve_fit
import sys

gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gL_FT  = lambda x,w: np.exp(-np.abs(x)*pi*w)
gE = lambda v,v0,w: np.exp(-np.abs(v-v0)/w)
corr_factor = lambda x,B,A,tau: A*np.exp(x/tau) + B

gE_FT = lambda x,w: 2*w / (1 + 4*pi**2*x**2*w**2) 

def calc_g_FT(x, w, g_FT, folding_thresh=1e-6):

    x_fold = (x, x[::-1])
    
    result = np.zeros(x.size)
    n = 0
    while g_FT(n/2,w) >= folding_thresh:
        result += g_FT(n/2 + x_fold[n&1], w)
        n += 1
##    result /= result[0]
    return result

vmax = 1000.0
dv = 0.1
w = 100.0

v_lin = np.arange(-vmax,vmax,dv)
N_lin = len(v_lin)
x_lin = rfftfreq(N_lin,1)
I_lin = gL(v_lin,0,w)

# Old calculation of Lorentzian:
I_g_FT = calc_g_FT(x_lin,w/dv,gL_FT)/dv


# Correction factor:
log_wvmax = np.log(w/vmax)
poptA = [ 6.74408847, 0.23299924, 0.53549119]
poptB = [ 6.82193751, 0.09226203, 0.49589094]
poptw = [-0.93309186, 0.18724358, 0.50806536]

A = w*np.exp(-corr_factor(log_wvmax,*poptA))
B = w*np.exp(-corr_factor(log_wvmax,*poptB))
w_corr = vmax*np.exp(corr_factor(log_wvmax,*poptw))

Err_corr = A*gE_FT(x_lin, w_corr/dv)*100/vmax**2
Err_corr[0] += B*200/vmax/dv
I_alternate = 1-(np.arange(len(x_lin))&1)*2
Err_corr *= I_alternate

I_g_IFT = ifftshift(irfft(I_g_FT))
I_g_FT -= Err_corr
I_g_IFT_corr = ifftshift(irfft(I_g_FT))

#Compare FT's:
plt.title('Fourier domain')
plt.plot(x_lin,rfft(fftshift(I_g_IFT-I_lin)).real * I_alternate,'k',label='DFT',lw=3)
plt.plot(x_lin,Err_corr * I_alternate,'r',label='Exact (corrected)')
plt.yscale('log')
plt.legend()
plt.show()

#Compare plots:
plt.title('Real domain')
plt.plot(v_lin, I_lin,'k',label='Direct',lw=3)
plt.plot(v_lin,I_g_IFT,'r--',label='IDFT naive')
plt.plot(v_lin,I_g_IFT_corr,'r',label='IDFT corrected')
plt.yscale('log')
plt.legend()
plt.show()
