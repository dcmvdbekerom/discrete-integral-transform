import numpy as np
from numpy import pi, log, exp, sqrt, cosh, sinh
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

coth = lambda x: cosh(x)/sinh(x)
csch = lambda x: 1/sinh(x)

gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gG  = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)
gE = lambda v,v0,w: np.exp(-np.abs(v-v0)/w)/(2*w)
gE_corr = lambda v,w,vmax: (exp((v-2*vmax)/w) + exp(-v/w))/(1-exp(-2*vmax/w))/(2*w)
dgLdv = lambda v,v0,w: -32/(pi*w**2) * ((v-v0)/w)/(1+4*((v-v0)/w)**2)**2

gL_FT  = lambda x,w: np.exp(-np.abs(x)*pi*w)
gG_FT  = lambda x,w: np.exp(-(x*pi*w)**2/(4*log(2)))
gE_FT = lambda x,w: 1 / (1 + 4*pi**2*x**2*w**2)

corr_fun = lambda x,C0,C1: C0 + C1*x**2

def init_w_axis(dx, log_wi):
    log_w_min = np.min(log_wi)
    log_w_max = np.max(log_wi) + 1e-4
    N = np.ceil((log_w_max - log_w_min)/dx) + 1
    log_w_arr = log_w_min + dx * np.arange(N)
    return log_w_arr
 
    
def get_indices(arr_i, axis):
    pos   = np.interp(arr_i, axis, np.arange(axis.size))
    index = pos.astype(int) 
    return index, index + 1, pos - index 


def calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i):
    
    S_klm = np.zeros((2 * v.size, log_wG.size, log_wL.size))

    ki0, ki1, avi = get_indices(v0i, v)          #Eqs 3.4 & 3.6
    li0, li1, aGi = get_indices(log_wGi, log_wG) #Eqs 3.7 & 3.10
    mi0, mi1, aLi = get_indices(log_wLi, log_wL) #Eqs 3.7 & 3.10

    np.add.at(S_klm, (ki0, li0, mi0), S0i * (1-avi) * (1-aGi) * (1-aLi))
    np.add.at(S_klm, (ki0, li0, mi1), S0i * (1-avi) * (1-aGi) *    aLi )
    np.add.at(S_klm, (ki0, li1, mi0), S0i * (1-avi) *    aGi  * (1-aLi))
    np.add.at(S_klm, (ki0, li1, mi1), S0i * (1-avi) *    aGi  *    aLi )
    np.add.at(S_klm, (ki1, li0, mi0), S0i *    avi  * (1-aGi) * (1-aLi))
    np.add.at(S_klm, (ki1, li0, mi1), S0i *    avi  * (1-aGi) *    aLi )
    np.add.at(S_klm, (ki1, li1, mi0), S0i *    avi  *    aGi  * (1-aLi))
    np.add.at(S_klm, (ki1, li1, mi1), S0i *    avi  *    aGi  *    aLi )
    
    return S_klm


def calc_gV_FT(x, wG, wL, folding_thresh=1e-6):

    gV_FT = lambda x,wG,wL: gG_FT(x,wG)*gL_FT(x,wL)
    x_fold = (x, x[::-1])
    
    result = np.zeros(x.size)
    n = 0
    while gV_FT(n/2,wG,wL) >= folding_thresh:
        result += gV_FT(n/2 + x_fold[n&1], wG, wL)
        n += 1

    return result


def apply_transform(Nv, log_wG, log_wL, S_klm, folding_thresh):
    x      = np.fft.rfftfreq(2 * Nv, 1)
    S_k_FT = np.zeros(Nv + 1, dtype = np.complex64)
    for l in range(log_wG.size):
        for m in range(log_wL.size):
            wG_l,wL_m = np.exp(log_wG[l]),np.exp(log_wL[m])
            gV_FT = calc_gV_FT(x, wG_l, wL_m, folding_thresh)    
            S_k_FT += np.fft.rfft(S_klm[:,l,m]) * gV_FT
    return np.fft.irfft(S_k_FT)[:Nv]


# Call I = synthesize_spectrum(v, v0i, log_wGi, log_wLi, S0i) to synthesize the spectrum.
# v = spectral axis (in cm-1)
# v0i = list/array with spectral position of lines
# log_wGi = list/array with log of the Gaussian widths of lines [= np.log(wGi)]
# log_wLi = list/array with log of the Lorentzian widths of lines [= np.log(wLi)]
# S0i = list/array with the linestrengths of lines (E.g. absorbances, emissivities, etc., depending the spectrum being synthesized)


def synthesize_spectrum(v, v0i, log_wGi, log_wLi, S0i, dxG = 0.14, dxL = 0.2, 
                        folding_thresh = 1e-6):
        
    idx = (v0i >= np.min(v)) & (v0i < np.max(v))
    v0i, log_wGi, log_wLi, S0i = v0i[idx], log_wGi[idx], log_wLi[idx], S0i[idx]

    log_dvi = np.interp(v0i, v[1:-1], np.log(0.5*(v[2:] - v[:-2])))
##    log_dvi = np.log(v0i * dxv) # log-grid
    
    log_wG = init_w_axis(dxG, log_wGi - log_dvi) #pts
    log_wL = init_w_axis(dxL, log_wLi - log_dvi) #pts
    
    S_klm = calc_matrix(v, log_wG, log_wL, v0i, log_wGi - log_dvi, log_wLi - log_dvi, S0i)
    
    dv = np.interp(v, v[1:-1], 0.5*(v[2:] - v[:-2]))
##    dv = v*dxv # log-grid
    
    I = apply_transform(v.size, log_wG, log_wL, S_klm, folding_thresh) / dv
    return I, S_klm

w_list = []
A_list = []
B_list = []

log_wvmax_arr = np.arange(-10,-2,0.05)
wvmax_arr = np.exp(log_wvmax_arr)
L = 5

Nv = 10000
vmax = 100
dv = vmax/Nv

##vmax = 100
##dv = 0.01
##Nv = int(vmax/dv)


k_arr = np.arange(Nv)
v_arr = np.arange(Nv)*dv
x = np.fft.rfftfreq(2 * Nv, 1)


for q in wvmax_arr:
    wL = q*vmax
    
##    wG = 0.0
##    wE = 3.0
    
##    IE_FT = gE_FT(x, wE/dv)/dv
##    IE_IDFT = np.fft.irfft(IE_FT)[:Nv]
##    IE_direct = gE(v_arr,0,wE)
##    IE_direct_corr = gE_corr(v_arr,wE,vmax)
##
##    plt.plot(v_arr, IE_IDFT, 'k',lw=3)
##    plt.plot(v_arr, IE_direct,'r--')
##    plt.plot(v_arr, IE_direct_corr, 'r-')
##    plt.yscale('log')
##    plt.show()

##    IL_FT = calc_gV_FT(x, 0, wL/dv)/dv
##    IL_IDFT = np.fft.irfft(IL_FT)[:Nv]
##    IL_direct = gL(v_arr,0,wL)
##    IL_err = IL_IDFT - IL_direct


    Ic0_arr = 2*gL(0,2*k_arr*vmax,wL)
    Ic1_arr = gL(vmax,2*k_arr*vmax,wL)+gL(vmax,-2*k_arr*vmax,wL)
    dIc1dv_arr = dgLdv(vmax,2*k_arr*vmax,wL)+dgLdv(vmax,-2*k_arr*vmax,wL)

    Ic0 = np.sum(Ic0_arr[1:])
    Ic1 = np.sum(Ic1_arr[1:])
    dIc1dv = np.sum(dIc1dv_arr[1:])

    IL_err = np.zeros(Nv)
    for n in range(1,L):
        IL_err += gL(v_arr,2*vmax*n,wL)
        IL_err += gL(v_arr,-2*vmax*n,wL)

    IL_err += 0.5*(np.sum(Ic0_arr[L:]) + np.sum(Ic1_arr[L:]))

    w0 = vmax/2
    A0 = 2*w0*(IL_err[-1]-IL_err[0])
    B0 = IL_err[0] - A0*gE_corr(vmax,w0,vmax)

    err_fun = lambda v,w,A,B: A*gE_corr(v,w,vmax)[::-1] + B/vmax

    popt = [w0,A0,B0]
    popt,pcov = curve_fit(err_fun,v_arr,IL_err,p0=popt)

    w_list.append(log(popt[0]/vmax))
    A_list.append(-log(popt[1]/q))
    B_list.append(-log(popt[2]/q))

##    plt.plot(v_arr, IL_err,'k',lw=3, label='err')
##    plt.plot(v_arr,err_fun(v_arr,*popt),'r-',label='fit')
##    plt.yscale('log')
##    plt.legend()
##    plt.show()

w_arr = np.array(w_list)
A_arr = np.array(A_list)
B_arr = np.array(B_list)

poptw, pcovw = curve_fit(corr_fun, wvmax_arr, w_arr)
poptA, pcovA = curve_fit(corr_fun, wvmax_arr, A_arr)
poptB, pcovB = curve_fit(corr_fun, wvmax_arr, B_arr)

##print(poptw[0]/-0.92817272, poptw[1]/0.19391863)
##print(poptA[0]/2.36500019, poptA[1]/0.06657851)
##print(poptB[0]/2.1889989232582137,poptB[1]/0.09044059608037858)

print(poptw)
print(poptA)
print(poptB)

plt.plot(log_wvmax_arr, w_arr - w_arr[0])
plt.plot(log_wvmax_arr, A_arr - A_arr[0])
plt.plot(log_wvmax_arr, B_arr - B_arr[0])

plt.plot(log_wvmax_arr, corr_fun(wvmax_arr, *poptw) - poptw[0], 'k--')
plt.plot(log_wvmax_arr, corr_fun(wvmax_arr, *poptA) - poptA[0], 'k--')
plt.plot(log_wvmax_arr, corr_fun(wvmax_arr, *poptB) - poptB[0], 'k--')

plt.yscale('log')
plt.show()








