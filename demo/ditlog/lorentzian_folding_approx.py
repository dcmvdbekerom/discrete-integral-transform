import numpy as np
from numpy import pi, log, exp, sqrt, cosh, sinh
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.fft import rfftfreq, rfft, irfft, fftshift, ifftshift 

gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gL_apx = lambda v,v0,w: 2/(pi*w) * 1 / (4*((v-v0)/w)**2)
gL_int = lambda v,v0,w: -1/pi * 1 / (2*(v-v0)/w)
dgLdv = lambda v,v0,w: -32/(pi*w**2) * ((v-v0)/w)/(1+4*((v-v0)/w)**2)**2


Nv = 1000
vmax = 10
dv = vmax/Nv

k_arr = np.arange(Nv)
v_arr = np.arange(Nv)*dv
x = rfftfreq(2 * Nv, 1)

wL = 3.4530


L = 100

IL_err = np.zeros(Nv)
for n in range(1,L):
    IL_err += gL(v_arr,2*vmax*n,wL)
    IL_err += gL(v_arr,-2*vmax*n,wL)


Ic0_arr = 2*gL(0,2*k_arr*vmax,wL)
Ic1_arr = gL(vmax,2*k_arr*vmax,wL)+gL(vmax,-2*k_arr*vmax,wL)
dIc1dv_arr = dgLdv(vmax,2*k_arr*vmax,wL)+dgLdv(vmax,-2*k_arr*vmax,wL)

Ic0 = np.sum(Ic0_arr[1:])
Ic1 = np.sum(Ic1_arr[1:])
dIc1dv = np.sum(dIc1dv_arr[1:])



IL_err0 = wL*pi/(24*vmax**2)
IL_err1 = wL/(2*pi*vmax**2)*(pi**2/4-1)
IL_err2 = 2*wL/(pi*vmax**3)


gE_corr = lambda v,w,vmax: (exp((v-2*vmax)/w) + exp(-v/w))/(2*w)


