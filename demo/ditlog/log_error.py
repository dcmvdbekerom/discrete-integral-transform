import numpy as np
import matplotlib.pyplot as plt
from numpy import pi,sqrt,log,exp
from numpy.fft import rfft,irfft,rfftfreq,fftshift,ifftshift

gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gG  = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)

dgLdv = lambda v,v0,w: -8*pi*(v-v0)/w * gL(v,v0,w)**2
dgGdv = lambda v,v0,w: -16/w**2*(v-v0)*log(2) * gG(v,v0,w)

##dgLdv = lambda v,v0,w: -32/(pi*w**2) * ((v-v0)/w)/(1+4*((v-v0)/w)**2)**2
##dgGdv = lambda v,v0,w: (-32/w**3)*(v-v0)*sqrt(log(2)**3/pi)*exp(-4*log(2)*((v-v0)/w)**2)

Nv = 10000
dv = 0.01 #cm-1
v0 = 2200.0 #cm-1
R = v0/dv

w = 5.0 #cm-1

k_arr = np.arange(-Nv//2,Nv//2)
v_lin = v0 + k_arr * dv
v_log = v0*exp(k_arr/R)

I_lin = gG(v_lin,v0,w)
I_log = gG(v_log,v0,w)
I_err = I_log - I_lin

plt.plot(v_lin[1:-1], (I_lin[2:] - I_lin[:-2])/dv)
plt.plot(v_lin, dgGdv(v_lin,v0,w),'k--')
plt.show()

plt.plot(k_arr,I_err,label = 'log-lin')

plt.plot(k_arr, 0.25/v0*(k_arr*dv)**2*dgGdv(v_lin,v0,w),'k--',label='predicted')
##plt.plot(k_arr, 0.25*v0*(v_lin/v0-1)**2*dgLdv(v_lin,v0,w),'k--',label='predicted')
##plt.plot(k_arr, 0.25*v0*((v_lin/v0)**2-2*(v_lin/v0)+1)*dgLdv(v_lin,v0,w),'k--',label='predicted')

##plt.plot(k_arr, 0.25*v0*(v_lin/v0)**2*dgLdv(v_lin,v0,w),'k',label='2')
##plt.plot(k_arr, 0.25*v0*-2*(v_lin/v0)*dgLdv(v_lin,v0,w),'--',label='1')
##plt.plot(k_arr, 0.25*v0*dgLdv(v_lin,v0,w),'-.',label='0')
plt.legend()
plt.show()
