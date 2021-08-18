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


def calc_gV_FT(x, wG, wL, folding_thresh):

    gV_FT = lambda x,wG,wL: np.exp(-((np.pi*x*wG)**2/(4*np.log(2)) + np.pi*x*wL))
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

    log_dvi = np.interp(v0i, v[1:-1], np.log(0.5*(v[2:] - v[:-2])))  # General
##    log_dvi = np.log(v0i * dxv) # log-grid
    
    log_wG = init_w_axis(dxG, log_wGi - log_dvi) #pts
    log_wL = init_w_axis(dxL, log_wLi - log_dvi) #pts
    
    S_klm = calc_matrix(v, log_wG, log_wL, v0i, log_wGi - log_dvi, log_wLi - log_dvi, S0i)
    
    dv = np.interp(v, v[1:-1], 0.5*(v[2:] - v[:-2])) # General
##    dv = v*dxv # log-grid
    
    I = apply_transform(v.size, log_wG, log_wL, S_klm, folding_thresh) / dv
    return I, S_klm
