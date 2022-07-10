import numpy as np
from numpy import pi, log, exp, sqrt, cosh, sinh
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

## function definitions:

coth = lambda x: cosh(x)/sinh(x)
csch = lambda x: 1/sinh(x)

gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gG  = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)
gE = lambda v,v0,w: exp(-np.abs(v-v0)/w)/(2*w)
gE_corr = lambda v,w,vmax: (exp((v-2*vmax)/w) + exp(-v/w))/(1-exp(-2*vmax/w))/(2*w)
dgLdv = lambda v,v0,w: -32/(pi*w**2) * ((v-v0)/w)/(1+4*((v-v0)/w)**2)**2

gL_FT = lambda x,w: exp(-np.abs(x)*pi*w)
gG_FT = lambda x,w: exp(-(x*pi*w)**2/(4*log(2)))
gV_FT = lambda x,wG,wL: gG_FT(x,wG)*gL_FT(x,wL)
gE_FT = lambda x,w: 1 / (1 + 4*pi**2*x**2*w**2)


##def calc_gV_FT(x, wG, wL, folding_thresh=1e-6):
##
##    gV_FT = lambda x,wG,wL: gG_FT(x,wG)*gL_FT(x,wL)
##
##    result = np.zeros(x.size)
##    n = 0
##    while gV_FT(n/2,wG,wL) >= folding_thresh:
##        result += gV_FT(n/2 + x[::1-2*(n&1)], wG, wL)
##        n += 1
##
##    return result

coeff_w = [0.39560962,-0.19461568]
coeff_A = [0.09432246, 0.06592025]
coeff_B = [0.11202818, 0.09048447]

corr_fun = lambda x,c0,c1: c0 * np.exp(-c1*x**2)

def gL_FT_corr(x_arr, wL):

    result = gL_FT(x_arr, wL)

    vmax = 1/(2*x_arr[1])
    q = wL/vmax
    
    w_corr = corr_fun(q, *coeff_w)*vmax
    A_corr = corr_fun(q, *coeff_A)*q
    B_corr = corr_fun(q, *coeff_B)*q

    I_corr = A_corr * gE_FT(x_arr, w_corr)
    I_corr[0] += 2*B_corr
    I_corr[1::2] *= -1

    result -= I_corr
    return result


## produce lineshapes:


Nv = 1000

dv = 0.01

v_arr = np.arange(-Nv,Nv)*dv
v_max = Nv*dv

x_arr = np.fft.rfftfreq(2*Nv, dv)

##wG = 0.0
wL = 1.0

# The lineshape we get when we calculate in real space:
I_direct = gL(v_arr,0,wL)
I_direct_FT = np.fft.rfft(np.fft.fftshift(I_direct*dv))

# The lineshape we get when we calculate in Fourier space, and invert:
I_FT = gL_FT(x_arr, wL)
I_IDFT = np.fft.ifftshift(np.fft.irfft(I_FT/dv))

# We can correct this by calculating the error and subtracting it:
Nk = 1000
I_IDFT_err = np.sum([gL(v_arr, 2*k*v_max, wL) + gL(v_arr, -2*k*v_max, wL) for k in range(1, Nk)], 0)
I_IDFT_corr = I_IDFT - I_IDFT_err
I_corr_FT = np.fft.rfft(np.fft.fftshift(I_IDFT_corr*dv))
##
##


fig, ax = plt.subplots(1,2)

ax[0].set_title('Real space')
ax[0].plot(v_arr, I_direct, 'k', lw=3, label = 'Direct')
ax[0].plot(v_arr, I_IDFT, 'r--', label='IDFT')
ax[0].set_yscale('log')
ax[0].plot(v_arr, I_IDFT_corr, 'y--', label='Corrected IDFT')
ax[0].legend()
ax[1].set_title('FT space')
ax[1].axhline(0,c='k',ls='--')
ax[1].axvline(0,c='k',ls='--')

ax[1].plot(x_arr, I_direct_FT, 'k-', lw=3, label='DFT')
ax[1].plot(x_arr, I_FT, 'r--', label='Direct FT')
ax[1].plot(x_arr, I_corr_FT, 'y--', label='Corrected DFT')

ax[1].set_yscale('log')
plt.xlabel('x / (2*v_max)')
ax[1].legend()
plt.savefig('fig2.png')
plt.show()
