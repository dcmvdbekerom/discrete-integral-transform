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


def calc_matrix(v, log_wL, v0i, log_wLi, S0i, zero_pad=2):
    
    S_klm = np.zeros((zero_pad * v.size, log_wL.size))

    ki0, ki1, avi = get_indices(v0i, v)          #Eqs 3.4 & 3.6
    mi0, mi1, aLi = get_indices(log_wLi, log_wL) #Eqs 3.7 & 3.10

    np.add.at(S_klm, (ki0, mi0), S0i * (1-avi) *   (1-aLi))
    np.add.at(S_klm, (ki0, mi1), S0i * (1-avi) *      aLi )
    np.add.at(S_klm, (ki1, mi0), S0i *    avi  *   (1-aLi))
    np.add.at(S_klm, (ki1, mi1), S0i *    avi  *      aLi )
    
    return S_klm


def calc_gV_FT(x, wL, folding_thresh):

    gV_FT = lambda x,wL: np.exp(-np.pi*x*wL)
    x_fold = (x, x[::-1])
    
    result = np.zeros(x.size)
    n = 0
    while gV_FT(n/2,wL) >= folding_thresh:
        result += gV_FT(n/2 + x_fold[n&1], wL)
        n += 1

    return result


def apply_transform(Nv, log_wL, S_klm, folding_thresh):
    x      = np.fft.rfftfreq(S_klm.shape[0], 1)
    S_k_FT = np.zeros(x.size, dtype = np.complex64)

    for m in range(log_wL.size):
        wL_m = np.exp(log_wL[m])
        gV_FT = calc_gV_FT(x, wL_m, folding_thresh)    
        S_k_FT += np.fft.rfft(S_klm[:,m]) * gV_FT
    return np.fft.irfft(S_k_FT)[:Nv]


def synthesize_spectrum(v, v0i, log_wLi, S0i, dxv, dxL = 0.2,
                        zero_pad = 2,
                        folding_thresh = 1e-12):
        
    idx = (v0i >= np.min(v)) & (v0i < np.max(v))
    v0i, log_wLi, S0i = v0i[idx], log_wLi[idx], S0i[idx]

    log_dvi = np.log(v0i * dxv)
    log_wL = init_w_axis(dxL, log_wLi - log_dvi) #pts
    
    S_klm = calc_matrix(v, log_wL, v0i, log_wLi - log_dvi, S0i, zero_pad)
    dv = v*dxv
    I = apply_transform(v.size, log_wL, S_klm, folding_thresh) / dv
    return I
