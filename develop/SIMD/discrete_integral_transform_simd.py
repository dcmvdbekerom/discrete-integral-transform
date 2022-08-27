## Simple implementation with core functionality but no error corrections

import numpy as np
from time import perf_counter
from py_simd.py_simd import add_at32

def init_w_axis(dx, log_wi):
    log_w_min = np.min(log_wi)
    log_w_max = np.max(log_wi) + 1e-4
    N = np.ceil((log_w_max - log_w_min)/dx) + 1
    log_w_arr = log_w_min + dx * np.arange(N)
    return log_w_arr
 
    
def get_indices(arr_i, axis):
    pos   = np.interp(arr_i, axis, np.arange(axis.size))
    index = pos.astype(int)
    a = (pos - index).astype(np.float32)
    return index, index + 1, a


## Calc matrix functions:

def calc_matrix_py1(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i):
    S_klm = np.zeros((2 * v.size, log_wG.size, log_wL.size), dtype=np.float32)
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
    
    
def calc_matrix_cy1(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i):
    S_klm = np.zeros((2 * v.size, log_wG.size, log_wL.size), dtype=np.float32)
    ki0, ki1, avi = get_indices(v0i, v)          #Eqs 3.4 & 3.6
    li0, li1, aGi = get_indices(log_wGi, log_wG) #Eqs 3.7 & 3.10
    mi0, mi1, aLi = get_indices(log_wLi, log_wL) #Eqs 3.7 & 3.10

    add_at32(S_klm, ki0, li0, mi0, S0i * (1-avi) * (1-aGi) * (1-aLi))
    add_at32(S_klm, ki0, li0, mi1, S0i * (1-avi) * (1-aGi) *    aLi )
    add_at32(S_klm, ki0, li1, mi0, S0i * (1-avi) *    aGi  * (1-aLi))
    add_at32(S_klm, ki0, li1, mi1, S0i * (1-avi) *    aGi  *    aLi )
    add_at32(S_klm, ki1, li0, mi0, S0i *    avi  * (1-aGi) * (1-aLi))
    add_at32(S_klm, ki1, li0, mi1, S0i *    avi  * (1-aGi) *    aLi )
    add_at32(S_klm, ki1, li1, mi0, S0i *    avi  *    aGi  * (1-aLi))
    add_at32(S_klm, ki1, li1, mi1, S0i *    avi  *    aGi  *    aLi )
    return S_klm


## Apply_transform functions:

def apply_transform_py1(v, log_wG, log_wL, S_klm):
    dv     = (v[-1] - v[0]) / (v.size - 1)
    x      = np.fft.rfftfreq(2 * v.size, dv)
    S_k_FT = np.zeros(v.size + 1, dtype = np.complex64)
    
    for l in range(log_wG.size):
        for m in range(log_wL.size):
            wG_l,wL_m = np.exp(log_wG[l]),np.exp(log_wL[m])
            gV_FT = np.exp(-((np.pi*wG_l*x)**2/(4*np.log(2)) + np.pi*wL_m*x))
            S_k_FT += np.fft.rfft(S_klm[:,l,m]) * gV_FT
            
    return np.fft.irfft(S_k_FT)[:v.size] / dv


## Synthesizie_spectrum function:

def synthesize_spectrum(v, v0i, log_wGi, log_wLi, S0i,
                        dxG = 0.1, dxL = 0.1,
                        f_calc_matrix=calc_matrix_py1,
                        f_apply_transform=apply_transform_py1):
    
    idx = (v0i >= np.min(v)) & (v0i < np.max(v))
    
    v0i = v0i[idx].astype(np.float32)
    log_wGi = log_wGi[idx].astype(np.float32)
    log_wLi = log_wLi[idx].astype(np.float32)
    S0i = S0i[idx].astype(np.float32)

    log_wG = init_w_axis(dxG,log_wGi) #Eq 3.8
    log_wL = init_w_axis(dxL,log_wLi) #Eq 3.9

    t_list = []
    t_list.append(perf_counter())
    
    S_klm = f_calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i)
    t_list.append(perf_counter())
    
    I = f_apply_transform(v, log_wG, log_wL, S_klm)
    t_list.append(perf_counter())
    
    return I, S_klm, t_list

if __name__ == '__main__':
    import compare_functions