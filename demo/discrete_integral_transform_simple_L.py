## Simple implementation with core functionality but no error corrections
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


def calc_matrix(v, log_wL, v0i, log_wLi, S0i):
    S_km = np.zeros((2 * v.size, log_wL.size))
    ki0, ki1, avi = get_indices(v0i, v)          #Eqs 3.4 & 3.6
    mi0, mi1, aLi = get_indices(log_wLi, log_wL) #Eqs 3.7 & 3.10
    
    np.add.at(S_km, (ki0, mi0), S0i * (1-avi) * (1-aLi))
    np.add.at(S_km, (ki0, mi1), S0i * (1-avi) *    aLi )
    np.add.at(S_km, (ki1, mi0), S0i *    avi  * (1-aLi))
    np.add.at(S_km, (ki1, mi1), S0i *    avi  *    aLi )
    return S_km


def apply_transform(v, log_wG, log_wL, S_klm):
    dv     = (v[-1] - v[0]) / (v.size - 1)
    x      = np.fft.rfftfreq(2 * v.size, dv)
    S_k_FT = np.zeros(v.size + 1, dtype = np.complex64)
    for m in range(log_wL.size):
        wL_m = np.exp(log_wL[m])
        gL_FT = np.exp(np.pi*wL_m*x))
        S_k_FT += np.fft.rfft(S_km[:,m]) * gL_FT
        
    return np.fft.irfft(S_k_FT)[:v.size] / dv


def synthesize_spectrum(v, v0i, log_wLi, S0i, dxL = 0.2):
    idx = (v0i >= np.min(v)) & (v0i < np.max(v))
    v0i, log_wLi, S0i = v0i[idx], log_wLi[idx], S0i[idx]
    log_wL = init_w_axis(dxL,log_wLi) #Eq 3.9
    S_km = calc_matrix(v, log_wL, v0i, log_wLi, S0i)
    I = apply_transform(v, log_wL, S_klm)
    return I, S_klm
