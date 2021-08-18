import numpy as np
import matplotlib.pyplot as plt
from ditlog import synthesize_spectrum as spectrum
from numpy.fft import rfftfreq, rfft, irfft, fftshift, ifftshift
from numpy import log,exp,pi,sqrt
from math import factorial

gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gG  = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)*dv
gL_FT  = lambda x,w: np.exp(-np.abs(x)*pi*w)
gG_FT  = lambda x,w: np.exp(-(x*pi*w)**2/(4*log(2)))


def calc_g_FT(x, w, g_FT, folding_thresh=1e-6):

    x_fold = (x, x[::-1])
    
    result = np.zeros(x.size)
    n = 0
    while g_FT(n/2,w) >= folding_thresh:
        result += g_FT(n/2 + x_fold[n&1], w)
        n += 1
##    result /= result[0]
    return result

vmin = 500.0
vmax = 1000.0



w = 50.0
g = gL
g_FT = gL_FT

dv = 1.0
dxv = dv/vmin

# Make spectral axis:
v_lin = np.arange(vmin,vmax,dv)

# Make one that is double the size to prevent circular convolution:
N_lin = len(v_lin)
v_lin2 = np.arange(-N_lin,N_lin,dv) + vmin
N_lin2 = len(v_lin2)

x_lin = rfftfreq(N_lin2,1)
I_lin = g(v_lin2,vmin,w)
I_lin_FT = rfft(fftshift(I_lin))
I_lin_IFT = ifftshift(irfft(I_lin_FT))
I_g_FT = calc_g_FT(x_lin,w/dv,g_FT)
I_g_IFT = ifftshift(irfft(I_g_FT))

###Compare DFT with FT:
##plt.plot(x_lin,I_lin_FT.real*dv,c='tab:blue', label= 'real')
##plt.plot(x_lin,I_lin_FT.imag*dv,c='tab:red', label= 'imag')
##plt.plot(x_lin,calc_g_FT(x_lin,w/dv,g_FT),'k--', label='exact')
##plt.legend()
##plt.show()

#Compare transformed with direct:
##plt.plot(v_lin2,I_lin_IFT, label='Real>DFT>IDFT')
##plt.plot(v_lin2,I_g_IFT, label='Fourier>IDFT')
##plt.plot(v_lin2,I_lin,'k--',label = 'Real')
Err_lin = I_g_IFT-I_lin
Err_lin_FT = 
plt.plot(v_lin2,I_g_IFT-I_lin,'k--',label = 'Real')
##plt.yscale('log')
plt.legend()
plt.show()

N_log = int(np.ceil(np.log(vmax/vmin)/dxv))
v_log = vmin*np.exp(np.arange(N_log)*dxv)
v_log2 = vmin*np.exp(np.arange(-N_log,N_log)*dxv)
N_log2 = len(v_log2)
dv_log = v_log*dxv
dv_log2 = v_log2*dxv

x_log = rfftfreq(N_log2,1)
I_log = g(v_log2,vmin,w)
I_log_FT = rfft(fftshift(I_log))
I_log_IFT = ifftshift(irfft(I_log_FT))

##plt.plot(x_log,I_log_FT.real*dv,c='tab:blue',label='real')
##plt.plot(x_log,I_log_FT.imag*dv,c='tab:red',label='imag')
##plt.plot(x_log,I_log_FT.imag/I_log_FT.real,c='tab:green')
##plt.plot(x_log,calc_g_FT(x_log,w/dv,g_FT), 'k--',label='exact re')
####plt.plot(x_log,g_FT(x_log,w/dv), 'k--',label='exact im')
##plt.legend()
##plt.show()

##plt.plot(v_log2,I_log_IFT, label='FT>IFT')
##plt.plot(v_log2,I_log,'k--',label = 'direct')
##plt.legend()
##plt.show()

##plt.plot(x_lin,I_lin_FT - g_FT(x_lin,w/dv)/dv)
##plt.plot(x_log,I_log_FT - g_FT(x_log,w/dv)/dv)
##plt.show()
