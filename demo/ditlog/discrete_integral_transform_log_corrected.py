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

##    result -= I_corr

    I_corr2 = np.sum(np.array([2*np.cos(4*k*pi*x_arr*vmax) for k in range(30)]),0)
    plt.plot(x_arr,I_corr2)
    plt.show()
    print(I_corr2.shape)
    result *= I_corr2

    return result

Nv = 10000
dv = 0.01

v_arr = np.arange(Nv)*dv
v_arr2 = np.arange(2*Nv)*dv
x_arr = np.fft.rfftfreq(2 * Nv, dv)
I_shift = 1-(np.arange(len(x_arr))&1)*2

wG = 0.0
wL = 4.0

IL_direct = gL(v_arr,0,wL)
IL_FT = calc_gV_FT(x_arr, 0, wL)
IL_IDFT = np.fft.irfft(IL_FT/dv)[:Nv]

##plt.plot(v_arr, IL_direct, 'k', lw=3)
##plt.plot(v_arr, IL_IDFT, 'r--')
##plt.yscale('log')
##plt.show()

IL_direct2 = np.fft.fftshift(gL(v_arr2,v_arr2[Nv],wL))
IL_direct_DFT = np.fft.rfft(IL_direct2)*dv.real
IL_err = IL_FT - IL_direct_DFT
##plt.plot(x_arr, IL_direct_DFT.real, 'k', lw=3)
##plt.plot(x_arr, IL_FT, 'r--')
##plt.yscale('log')
##plt.show()

##vmax = Nv*dv
vmax = 1/(2*x_arr[1])
q = wL/vmax
##q = wL*x_arr[1]

w_corr = corr_fun(q, *coeff_w)*vmax
A_corr = corr_fun(q, *coeff_A)*q
B_corr = corr_fun(q, *coeff_B)*q

IL_FT_corr = A_corr * gE_FT(x_arr, w_corr) #dv = 1/(x_arr[1]*Nv)
IL_FT_corr[0] += 2*B_corr

##plt.plot(x_arr, IL_err*I_shift, 'k', lw=3)
##plt.plot(x_arr, IL_FT_corr, 'r')
##plt.yscale('log')
##plt.show()

IL_FT_corr[1::2] *= -1
IL_FT -= IL_FT_corr

##IL_FT *= IL_FT_corr2




IL_FT = gL_FT_corr(x_arr, wL)
##IL_FT = gL_FT_corr(x_arr*dv, wL/dv)
IL_IDFT_corr = np.fft.irfft(IL_FT/dv)[:Nv]

plt.plot(v_arr, IL_direct/IL_direct[0], 'k', lw=3)
plt.plot(v_arr, IL_IDFT/IL_IDFT[0], 'r--')
plt.plot(v_arr, IL_IDFT_corr/IL_IDFT_corr[0], 'r-')
plt.yscale('log')
plt.show()








