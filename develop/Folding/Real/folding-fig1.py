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
Nk = 1000
dv = 0.01

v_arr = np.arange(Nv)*dv
v_max = Nv*dv

x_arr = np.fft.rfftfreq(2 * Nv, dv)

##wG = 0.0
wL = 1.0

# The lineshape we get when we calculate in real space:
IL_direct = gL(v_arr,0,wL)

# The lineshape we get when we calculate in Fourier space, and invert:
IL_FT = gL_FT(x_arr, wL)
IL_IDFT = np.fft.irfft(IL_FT/dv)[:Nv]

# We can correct this by calculating the error and subtracting it:
IL_IDFT_err = np.sum([gL(v_arr, 2*k*v_max, wL) + gL(v_arr, -2*k*v_max, wL) for k in range(1, Nk)], 0)
IL_IDFT_corr = IL_IDFT - IL_IDFT_err

##
##B0 = wL*np.pi/(24*v_max**2) + gL(0,0,wL)
##plt.axhline(B0, c='r', ls='--')
##print(B0)

plt.title('Real space')
plt.plot(v_arr, IL_direct, 'k', lw=3, label = 'Direct')
plt.plot(v_arr, IL_IDFT, 'b-', label='IDFT')
plt.plot(v_arr, IL_IDFT_corr, 'r--', label='corrected IDFT')
plt.yscale('log')
plt.xlabel('v / wL')
plt.legend()
plt.savefig('fig1.png')
plt.show()
