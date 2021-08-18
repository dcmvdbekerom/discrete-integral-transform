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

vmax = 1000.0

w = 1.0
g = gL
g_FT = gL_FT

dv = 0.1

v_lin = np.arange(-vmax,vmax,dv)
N_lin = len(v_lin)
x_lin = rfftfreq(N_lin,1)

I_lin = g(v_lin,0,w)
I_lin_FT = rfft(fftshift(I_lin))
I_lin_IFT = ifftshift(irfft(I_lin_FT))

I_g_FT = calc_g_FT(x_lin,w/dv,g_FT)
I_g_IFT = ifftshift(irfft(I_g_FT/dv))

I_corr = np.zeros(N_lin)
Ic_list = [2*g(2**k*vmax,0,w) for k in range(100)]
Ic_arr = np.cumsum(Ic_list[::-1])[::-1]
for k in range(1,8):
    print(k)
    v_ex = np.arange(-2**k*vmax,2**k*vmax,dv)
    N_ex = len(v_ex)
    x_ex = rfftfreq(N_ex,1)
    I_ex_FT = calc_g_FT(x_ex,w/dv,g_FT)/dv
    I_ex_IFT = ifftshift(irfft(I_ex_FT))
    I_corr_k = np.roll(I_ex_IFT,N_lin//2)[:N_lin]
    I_corr += I_corr_k
    plt.plot(v_lin,I_corr_k,label=k)
    plt.plot(v_lin,Ic_arr[k]*np.ones(len(v_lin)),'k--')

##for k in range(1,5):
##    v_ex = np.arange(-2**k*vmax,2**k*vmax,dv)
##    N_ex = len(v_ex)
##    I_ex_FT = calc_g_FT(x_lin,(w/dv)/2**k,g_FT)/dv/2**k
##    I_ex_IFT = ifftshift(irfft(I_ex_FT))
##    I_corr_k = np.roll(I_ex_IFT,N_lin//2)[:N_lin]
##    plt.plot(v_lin*2**k,I_corr_k,'k--')


##plt.yscale('log')
plt.show()

###Compare DFT with FT:
##plt.plot(x_lin,I_lin_FT.real*dv,c='tab:blue', label= 'real')
##plt.plot(x_lin,I_lin_FT.imag*dv,c='tab:red', label= 'imag')
##plt.plot(x_lin,calc_g_FT(x_lin,w/dv,g_FT),'k--', label='exact')
##plt.legend()
##plt.show()


#####Compare transformed with direct:
plt.plot(v_lin,I_lin_IFT, label='Real>DFT>IDFT')

plt.plot(v_lin,I_g_IFT, label='Fourier>IDFT')
plt.plot(v_lin,I_g_IFT-I_corr, '-.',label='Fourier>IDFT [correctd]')
plt.plot(v_lin,I_lin,'k--',label = 'Real')

plt.yscale('log')
plt.legend()
plt.show()
