import numpy as np
import matplotlib.pyplot as plt
from numpy import pi,sqrt,log,exp
from numpy.fft import rfft,irfft,rfftfreq,fftshift,ifftshift

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
    result =  1j/(2*pi*v0*x) * (-2*zG**2 - 2*zG*zL + 3*zG - 0.5*zL**2 + zL) * gV_FT
    result[0] = 0
    return result

Nv = 100000
R = 100000
v0 = 2000.0 #cm-1
dv = v0/R

w = 1.0 #cm-1

k_arr = np.arange(-Nv//2,Nv//2)
x_arr = np.arange(0,Nv//2+1)/(Nv//2) * 1/(2*dv)

v_lin = v0 + k_arr * dv
v_log = v0*exp(k_arr/R)

I_lin = gL(v_lin,v0,w)
I_log = gL(v_log,v0,w)
I_err = I_log - I_lin

#### Derivative:
##plt.plot(v_lin[1:-1], 0.5*(I_lin[2:] - I_lin[:-2])/dv)
##plt.plot(v_lin, dgLdv(v_lin,v0,w),'k--')
##plt.show()
##
### Error:
##plt.plot(k_arr,I_err,label = 'log-lin')
##plt.plot(k_arr, 1/(2*v0)*(v_lin-v0)**2*dgLdv(v_lin,v0,w),'k--')
##plt.legend()
##plt.show()
##
### Check if FFT works:
##I_lin_FT = rfft(fftshift(I_lin)).real*dv
##plt.plot(x_arr,I_lin_FT)
##plt.plot(x_arr,gL_FT(x_arr,w),'k--')
##plt.show()

### Compare error with exact in FT domain:
##I_err_FT = rfft(fftshift(I_err)).imag*dv
##plt.plot(x_arr,I_err_FT)
##plt.plot(x_arr,g_err_FT(x_arr,v0,0,w),'k--')
##plt.show()

## Check reverse transform:
I_err_FT = g_err_FT(x_arr,v0,0,w)/dv
I_err_IDFT = ifftshift(irfft(I_err_FT))
##plt.plot(k_arr, I_lin)
##plt.plot(k_arr, I_log)
plt.plot(k_arr, I_err)
plt.plot(k_arr, I_err_IDFT,'k--')
plt.show()



