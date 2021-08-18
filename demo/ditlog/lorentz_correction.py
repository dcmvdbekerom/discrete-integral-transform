import numpy as np
import matplotlib.pyplot as plt
from ditlog import synthesize_spectrum as spectrum
from numpy.fft import rfftfreq, rfft, irfft, fftshift, ifftshift
from numpy import log,exp,pi,sqrt
from math import factorial
import sys


gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gG  = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)*dv
gL_FT  = lambda x,w: np.exp(-np.abs(x)*pi*w)
gG_FT  = lambda x,w: np.exp(-(x*pi*w)**2/(4*log(2)))
gC = lambda v,w: np.exp(-v/w)
dgLdv = lambda v,v0,w: -32/(pi*w**2) * ((v-v0)/w)/(1+4*((v-v0)/w)**2)**2
fitfun = lambda v,w,A,B: A*(exp((np.abs(v)-2*vmax)/w) + exp(-np.abs(v)/w))/(1-exp(-2*vmax/w)) + B
absexp = lambda v,v0,w: exp(-np.abs(v-v0)/w)

coth = lambda x: np.cosh(x)/np.sinh(x)
csch = lambda x: 1/np.sinh(x)

def calc_g_FT(x, w, g_FT, folding_thresh=1e-6):

    x_fold = (x, x[::-1])
    
    result = np.zeros(x.size)
    n = 0
    while g_FT(n/2,w) >= folding_thresh:
        result += g_FT(n/2 + x_fold[n&1], w)
        n += 1
##    result /= result[0]
    return result

vmax = 10.0

w = 5.0
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
I_g_IFT =ifftshift(irfft(I_g_FT/dv))

##plt.plot(v_lin[1:-1],(I_lin[2:]-I_lin[:-2])/dv)
##plt.plot(v_lin,dgLdv(v_lin,0,w),'k--')
##plt.show()
##sys.exit()


I_res = absexp(v_lin,0,w)
for k in range(1,4):
    I_res += absexp(v_lin,2*vmax*k,w)
    I_res += absexp(v_lin,-2*vmax*k,w)

##plt.plot(v_lin,I_res)
##plt.plot(v_lin,fitfun(v_lin,w),'k--')
##plt.show()
##sys.exit()






k_arr = np.arange(100)
Ic0_arr = 2*g(0,2*k_arr*vmax,w)
Ic1_arr = g(vmax,2*k_arr*vmax,w)+g(vmax,-2*k_arr*vmax,w)
dIc1dv_arr = dgLdv(vmax,2*k_arr*vmax,w)+dgLdv(vmax,-2*k_arr*vmax,w)

Ic0 = np.sum(Ic0_arr[1:])
Ic1 = np.sum(Ic1_arr[1:])
dIc1dv = np.sum(dIc1dv_arr[1:])


L = 5
I_corr = np.zeros(N_lin)
for n in range(1,L):
    I_corr += g(v_lin,2*vmax*n,w)
    I_corr += g(v_lin,-2*vmax*n,w)

I_corr += 0.5*(np.sum(Ic0_arr[L:]) + np.sum(Ic1_arr[L:]))

##    plt.plot(v_lin,I_corr,c='tab:blue')
##    plt.plot([-vmax,0,vmax],[Ic1[n],Ic0[n],Ic1[n]],'ko')
##    plt.plot(v_lin, 0.5*dIc1dv[n]*(v_lin-vmax)+Ic1[n])
plt.plot(v_lin,I_corr,c='tab:blue')

B = Ic0
A = Ic1 - B

w_fit = -(Ic0 - Ic1)/dIc1dv
A = (Ic1 - Ic0)
B = Ic1 + A*coth(-vmax/w_fit)

plt.plot(v_lin,np.roll(fitfun(v_lin,w,A,B),N_lin//2),'k--')

##plt.yscale('log')
plt.show()
sys.exit()

##for n in range(2,10):
##    I_corr = np.zeros(N_lin)
##    for k in range(1,n):
##        I_corr += g(v_lin,2*vmax*k,w)
##        I_corr += g(v_lin,-2*vmax*k,w)
##
##    plt.plot(v_lin,I_corr+Ic_arr[n])



##plt.plot(v_lin,gC(v_lin,1.0))
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
