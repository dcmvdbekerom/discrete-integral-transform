import numpy as np
import matplotlib.pyplot as plt
from numpy import pi,sqrt,log,exp
from numpy.fft import rfft,irfft,rfftfreq,fftshift,ifftshift
import discrete_integral_transform_log as dl

gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gG  = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)

gL_FT  = lambda x,w: np.exp(-np.abs(x)*pi*w)
gG_FT  = lambda x,w: np.exp(-(x*pi*w)**2/(4*log(2)))

dgLdv = lambda v,v0,w: -4*pi*(v-v0)/w * gL(v,v0,w)**2
dgGdv = lambda v,v0,w: -8/w**2*(v-v0)*log(2) * gG(v,v0,w)

def g_err_FT(x,v0,wG,wL):
    zG = (x*pi*wG)**2/(4*log(2))
    zL = np.abs(x)*pi*wL
    gV_FT = gG_FT(x,wG) * gL_FT(x,wL)
    result =  1j/(2*pi*x) * (2*zG**2 + 2*zG*zL - 3*zG + 0.5*zL**2 - zL) * gV_FT
    result[0] = 0
    return result

Nv = 100000
R = 100000
v0 = 2000.0 #cm-1
dv = v0/R

wG = 0.0
wL = 1.0 #cm-1

k_arr = np.arange(-Nv//2,Nv//2)
x_arr = np.arange(0,Nv//2+1)/(Nv//2) * 1/(2*dv)

v_lin = v0 + k_arr * dv
v_log = v0*exp(k_arr/R)

I_lin = gL(v_lin,v0,wL)
I_lin_IDFT = dl.gV(v_lin, v0, wG, wL)
I_lin_err = I_lin_IDFT - I_lin

I_log = gL(v_log,v0,wL)
I_log_IDFT = dl.gV(v_log, v0, wG, wL, log_correction=False)
I_log_err = I_log_IDFT - I_log


I_err = I_lin - I_log
Iv_err_FT = g_err_FT(x_arr,v0,0,wL)/dv
I_err_IDFT = ifftshift(irfft(Iv_err_FT))/v_log

I_log_corr = I_log_IDFT - I_err_IDFT
I_log_corr_err = I_log_corr - I_log

I_log_corr_IDFT = dl.gV(v_log, v0, wG, wL, log_correction=True)
I_log_corr_IDFT_err = I_log_corr_IDFT - I_log


##plt.plot(k_arr, I_lin_err, '-', label='lin_FT - lin')
##plt.plot(k_arr, I_log_IDFT - I_lin, '-', label='log_FT - lin')
# Conclusion: The naive FT result is the same as a lineshape produced by v_lin 

##plt.plot(k_arr, I_err, '-', label='err')
##plt.plot(k_arr, I_log_err, '-', label='log_FT - log')
##plt.plot(k_arr, I_err_IDFT, '-', label='err_IDFT')
# Conclusion: The error is very well reproduced

##plt.plot(k_arr, I_log_err, label='log_FT - log')
##plt.plot(k_arr, I_log_corr_err, label='log_FT_corr - log')
# Conclusion: Correction reduces error by 2-3 orders of magnitude

plt.plot(k_arr, I_log_corr_err, label='this script')
plt.plot(k_arr, I_log_corr_IDFT_err, label='lib')

plt.xlim(-60,60)
##plt.yscale('log')
plt.legend()
plt.show()






##plt.plot(k_arr, I_err, '.-', label='log-lin')
##plt.plot(k_arr, I_err2, '.-', label='log_FT-lin')




##
#### Check reverse transform:
##Iv_err_FT = g_err_FT(x_arr,0,wL)/dv
##I_err_IDFT = ifftshift(irfft(Iv_err_FT))/v_log
####plt.plot(v_log, I_log, label='Exact')
##





##plt.yscale('log')


##plt.plot(v_log, I_lin + I_err_IDFT, 'k--', label='FT, corrected')




