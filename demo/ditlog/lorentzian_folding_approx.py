import numpy as np
from numpy import pi, log, exp, sqrt, cosh, sinh, tanh, cos, arccos
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.fft import rfftfreq, rfft, irfft, fftshift, ifftshift
import sys

gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gL_apx = lambda v,v0,w: 2/(pi*w) * 1 / (4*((v-v0)/w)**2)
gL_int = lambda v,v0,w: -1/pi * 1 / (2*(v-v0)/w)
dgLdv = lambda v,v0,w: -16/(pi*w**2) * ((v-v0)/w)/(1+4*((v-v0)/w)**2)**2
gE_corr = lambda v,w,vmax: (exp((v-vmax)/w) + exp((-v-vmax)/w))/(2*w)
gE = lambda v,w,vmax: exp((v-vmax)/w)
fitfun = lambda v,A,B,w: A*gE_corr(v,w,vmax) + B/vmax


Nv = 1000
vmax = 10000
dv = vmax/Nv

k_arr = np.arange(Nv)
v_arr = np.arange(Nv)*dv
x = rfftfreq(2 * Nv, 1)

wL = 1.0


L = 10000

IL_err = np.zeros(Nv)
for n in range(1,L):
    IL_err += gL(v_arr,2*vmax*n,wL)
    IL_err += gL(v_arr,-2*vmax*n,wL)


IL_apx = np.zeros(Nv)
for n in range(1,L):
    IL_apx += gL_apx(v_arr,2*vmax*n,wL)
    IL_apx += gL_apx(v_arr,-2*vmax*n,wL)
    

f_0 = wL*pi/(24*vmax**2)
f_vmax = wL/(2*pi*vmax**2)*(pi**2/4-1)
dfdv_vmax = wL/(pi*vmax**3)

##t_arr = np.arange(0,10,0.01)
##rt_tanht = tanh(t_arr)/t_arr
##apx1 = 1+2*(cos(t_arr)-1)/3
##
##plt.plot(t_arr,rt_tanht)
##plt.plot(t_arr,apx1)
##plt.show()
##sys.exit()

q0 = (f_vmax - f_0)/(dfdv_vmax*vmax)
q_old = q0
for i in range(10):
    q_new = q0/tanh(1/(2*q_old))
    print(q_old)
    if not np.abs(q_new - q_old):
        break
    q_old = q_new

wc = q_new * vmax
Ac = 2*wc**2*dfdv_vmax/(1-exp(-2*vmax/wc))
Bc = (f_0 - Ac*exp(-vmax/wc)/wc)*vmax

plt.plot(v_arr,IL_err,'k',lw=3,label='direct')
plt.plot(v_arr,IL_apx,'r',lw=1,label='approx')

plt.plot([0,vmax],[f_0,f_vmax],'bo',label='exact')
plt.plot(v_arr,(v_arr - vmax) * dfdv_vmax + f_vmax,'b--')


plt.plot(v_arr,fitfun(v_arr,Ac,Bc,wc),'g-.',label='fit')

plt.legend()
plt.show()






