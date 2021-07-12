import numpy as np


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


def calc_matrix(log_v, log_wG, log_wL, log_v0i, log_wGi, log_wLi, S0i):

    S_klm = np.zeros((2 * log_v.size, log_wG.size, log_wL.size))

    ki0, ki1, avi = get_indices(log_v0i, log_v)  #Eqs 3.4 & 3.6
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


def calc_gV_FT(x, wG, wL):
    gG_FT = np.exp(-(np.pi*x*wG)**2/(4*np.log(2)))
    gL_FT = np.exp(- np.pi*x*wL)
    return gG_FT * gL_FT


def apply_transform(Nv, log_wG, log_wL, S_klm, folding_thresh):

    x      = np.fft.rfftfreq(2 * Nv, 1)
    S_k_FT = np.zeros(Nv + 1, dtype = np.complex64)
    for l in range(log_wG.size):
        for m in range(log_wL.size):
            wG_l,wL_m = np.exp(log_wG[l]),np.exp(log_wL[m])
            gV_FT = calc_gV_FT(x,wG_l,wL_m) 
            n = 1
            while calc_gV_FT(n/2,wG_l,wL_m) >= folding_thresh:
                gV_FT += calc_gV_FT(n/2 + x[::1-2*(n&1)],wG_l,wL_m)
                n += 1           
            S_k_FT += np.fft.rfft(S_klm[:,l,m]) * gV_FT
    return np.fft.irfft(S_k_FT)[:Nv]


# Call I = synthesize_spectrum(v, v0i, log_wGi, log_wLi, S0i) to synthesize the spectrum.
# v = spectral axis (in cm-1)
# v0i = list/array with spectral position of lines
# log_wGi = list/array with log of the Gaussian widths of lines [= np.log(wGi)]
# log_wLi = list/array with log of the Lorentzian widths of lines [= np.log(wLi)]
# S0i = list/array with the linestrengths of lines (E.g. absorbances, emissivities, etc., depending the spectrum being synthesized)


def synthesize_spectrum(log_v0i, log_wGi, log_wLi, S0i,
                        log_v_min, Nv, dxv, dxG = 0.14, dxL = 0.2, 
                        folding_thresh = 1e-6):
    
    log_v = log_v_min + np.arange(Nv) * dxv
    idx = (log_v0i >= np.min(log_v)) & (log_v0i < np.max(log_v))
    log_v0i, log_wGi, log_wLi, S0i = log_v0i[idx], log_wGi[idx], log_wLi[idx], S0i[idx]
    
    log_wG = init_w_axis(dxG, log_wGi - log_v0i - np.log(dxv)) #pts
    log_wL = init_w_axis(dxL, log_wLi - log_v0i - np.log(dxv)) #pts
    
    S_klm = calc_matrix(log_v, log_wG, log_wL,
                        log_v0i,
                        log_wGi  - log_v0i - np.log(dxv),
                        log_wLi  - log_v0i - np.log(dxv),
                        S0i)
    
    I = apply_transform(Nv, log_wG, log_wL, S_klm, folding_thresh)
    I/= np.exp(log_v) * dxv
    return log_v, I, S_klm