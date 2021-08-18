import numpy as np
import matplotlib.pyplot as plt
from ditlog import synthesize_spectrum as spectrum
from numpy.fft import rfftfreq, rfft, irfft, fftshift, ifftshift
from numpy import log,exp,pi,sqrt
from math import factorial
from scipy.optimize import curve_fit
import sys


gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gG  = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)*dv
gL_FT  = lambda x,w: np.exp(-np.abs(x)*pi*w)
gG_FT  = lambda x,w: np.exp(-(x*pi*w)**2/(4*log(2)))
gC = lambda v,w: np.exp(-v/w)
dgLdv = lambda v,v0,w: -32/(pi*w**2) * ((v-v0)/w)/(1+4*((v-v0)/w)**2)**2
fitfun = lambda v,w,A,B: np.roll(A*(exp((np.abs(v)-2*vmax)/w) + exp(-np.abs(v)/w))/(1-exp(-2*vmax/w)) + B,N_lin//2)
absexp = lambda v,v0,w: exp(-np.abs(v-v0)/w)
fitfun2 = lambda x,B,A,tau: A*np.exp(x/tau) + B

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
dv = 0.1

L = 3

v_lin = np.arange(-vmax,vmax,dv)
N_lin = len(v_lin)

k_arr = np.arange(2000)
log_wvmax_arr = np.arange(-10,0.05,0.05)

A_list = []
B_list = []
w_list = []

for log_wvmax in log_wvmax_arr:
    print(log_wvmax)
    w = np.exp(log_wvmax)*vmax
    I_lin = gL(v_lin,0,w)

    Ic0_arr = 2*gL(0,2*k_arr*vmax,w)
    Ic1_arr = gL(vmax,2*k_arr*vmax,w)+gL(vmax,-2*k_arr*vmax,w)
    dIc1dv_arr = dgLdv(vmax,2*k_arr*vmax,w)+dgLdv(vmax,-2*k_arr*vmax,w)

    Ic0 = np.sum(Ic0_arr[1:])
    Ic1 = np.sum(Ic1_arr[1:])
    dIc1dv = np.sum(dIc1dv_arr[1:])


    I_corr = np.zeros(N_lin)
    for n in range(1,L):
        I_corr += gL(v_lin,2*vmax*n,w)
        I_corr += gL(v_lin,-2*vmax*n,w)

    I_corr += 0.5*(np.sum(Ic0_arr[L:]) + np.sum(Ic1_arr[L:]))

    w_fit = -(Ic0 - Ic1)/dIc1dv
    A = (Ic1 - Ic0)
    B = Ic1 + A*coth(-vmax/w_fit)

    popt = [w_fit,A,B]
    popt,pcov = curve_fit(fitfun,v_lin,I_corr,p0=popt)
    w_fit,A,B = popt

    A_list.append(np.log(w/A))
    B_list.append(np.log(w/B))
    w_list.append(np.log(w_fit/vmax))
##
##    plt.plot(v_lin,I_corr,c='tab:blue')
##    plt.plot(v_lin,fitfun(v_lin,*popt),'k--')
##    plt.show()
    
A_arr = np.array(A_list)
B_arr = np.array(B_list)
w_arr = np.array(w_list)




poptA, pcov = curve_fit(fitfun2,log_wvmax_arr,A_arr)
poptB, pcov = curve_fit(fitfun2,log_wvmax_arr,B_arr)
poptw, pcov = curve_fit(fitfun2,log_wvmax_arr,w_arr)

plt.plot(log_wvmax_arr, A_arr, label='A')
plt.plot(log_wvmax_arr, fitfun2(log_wvmax_arr,*poptA), 'k--')
plt.plot(log_wvmax_arr, B_arr, label='B')
plt.plot(log_wvmax_arr, fitfun2(log_wvmax_arr,*poptB), 'k--')
plt.plot(log_wvmax_arr, w_arr, label='w')
plt.plot(log_wvmax_arr, fitfun2(log_wvmax_arr,*poptw), 'k--')
plt.legend()
plt.show()

plt.plot(log_wvmax_arr, A_arr-poptA[0], label='A')
plt.plot(log_wvmax_arr, fitfun2(log_wvmax_arr,*poptA)-poptA[0], 'k--')
plt.plot(log_wvmax_arr, B_arr-poptB[0], label='B')
plt.plot(log_wvmax_arr, fitfun2(log_wvmax_arr,*poptB)-poptB[0], 'k--')
plt.plot(log_wvmax_arr, w_arr-poptw[0], label='w')
plt.plot(log_wvmax_arr, fitfun2(log_wvmax_arr,*poptw)-poptw[0], 'k--')
plt.legend()
plt.yscale('log')
plt.show()

print(poptA)
print(poptB)
print(poptw)
