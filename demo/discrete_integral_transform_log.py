import numpy as np
from numpy import pi, log

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


gE_FT = lambda x,w: 1 / (1 + 4*pi**2*x**2*w**2)
gL_FT = lambda x,w: np.exp(-np.abs(x)*pi*w)
gG_FT = lambda x,w: np.exp(-(x*pi*w)**2/(4*log(2)))
gV_FT = lambda x,wG,wL: gG_FT(x,wG) * gL_FT(x,wL)

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

    result -= I_corr

    return result

gV_FT_corr = lambda x,wG,wL: gG_FT(x,wG) * gL_FT_corr(x,wL)


def calc_gV_FT(x, wG, wL, folding_thresh):
    
    result = gV_FT_corr(x, wG, wL)
    dv = 1/(2*x[-1])
    n = 1
    while gV_FT(n/(2*dv),wG,wL) >= folding_thresh:
        result += gV_FT(n/(2*dv) + x[::1-2*(n&1)], wG, wL)
        n += 1

    return result


def apply_transform(v, log_wG, log_wL, S_klm, folding_thresh, log_correction):
    Nv = v.size
    x = np.fft.rfftfreq(2 * Nv, 1)
    S_k_FT_re = np.zeros(Nv + 1, dtype = np.complex64)
    Sv_k_FT_im = np.zeros(Nv + 1, dtype = np.complex64)
    for l in range(log_wG.size):
        for m in range(log_wL.size):
            wG_l,wL_m = np.exp(log_wG[l]),np.exp(log_wL[m])
            gV_FT = calc_gV_FT(x, wG_l, wL_m, folding_thresh)
            
            S_klm_FT = np.fft.rfft(S_klm[:,l,m]) * gV_FT
            S_k_FT_re += S_klm_FT

            if log_correction:
                WGx = (pi*wG_l*x)**2/(4*log(2))
                WLx = pi*wL_m*x
                S_klm_FT *= (-2*WGx**2 + 3*WGx - 2*WGx*WLx - 0.5*WLx**2 + WLx)
                Sv_k_FT_im += 1j/(2*pi*x) * S_klm_FT
            
    S_k = np.fft.irfft(S_k_FT_re)[:Nv]
    if log_correction:
        Sv_k_FT_im[0] = 0
        dv = np.interp(v, v[1:-1], 0.5*(v[2:] - v[:-2]))
        S_k += np.fft.irfft(Sv_k_FT_im)[:Nv] * dv / v
        
    return S_k


def gV(v, v0, wG, wL, log_correction=False):
    dv0 = np.interp(v0, v[1:-1], 0.5*(v[2:] - v[:-2]))
    S_klm = np.zeros((2 * v.size, 1, 1))
    ki0, ki1, avi = get_indices([v0], v)
    S_klm[ki0[0],0,0] = (1-avi[0])/dv0
    S_klm[ki1[0],0,0] = avi[0]/dv0
    
    dvi = np.interp(v, v[1:-1], 0.5*(v[2:] - v[:-2]))
    return apply_transform(v,
                           np.array([log(wG/dv0)]), np.array([log(wL/dv0)]),
                           S_klm, 1e-6, log_correction)


# Call I = synthesize_spectrum(v, v0i, log_wGi, log_wLi, S0i) to synthesize the spectrum.
# v = spectral axis (in cm-1)
# v0i = list/array with spectral position of lines
# log_wGi = list/array with log of the Gaussian widths of lines [= np.log(wGi)]
# log_wLi = list/array with log of the Lorentzian widths of lines [= np.log(wLi)]
# S0i = list/array with the linestrengths of lines (E.g. absorbances, emissivities, etc., depending the spectrum being synthesized)


def synthesize_spectrum(v, v0i, log_wGi, log_wLi, S0i, dxG = 0.14, dxL = 0.2, 
                        folding_thresh = 1e-6, log_correction=False):
        
    idx = (v0i >= np.min(v)) & (v0i < np.max(v))
    v0i, log_wGi, log_wLi, S0i = v0i[idx], log_wGi[idx], log_wLi[idx], S0i[idx]

    log_dvi = np.interp(v0i, v[1:-1], np.log(0.5*(v[2:] - v[:-2])))  # General
    dvi = np.exp(log_dvi)
    
    log_wG = init_w_axis(dxG, log_wGi - log_dvi) #pts
    log_wL = init_w_axis(dxL, log_wLi - log_dvi) #pts
    
    S_klm = calc_matrix(v, log_wG, log_wL, v0i, log_wGi - log_dvi, log_wLi - log_dvi, S0i)

    dv = np.interp(v, v[1:-1], 0.5*(v[2:] - v[:-2]))
    I = apply_transform(v, log_wG, log_wL, S_klm,
                        folding_thresh, log_correction)/dv
    return I, S_klm


def gV_debug(v, v0, wG, wL, folding_thresh=1e-6, log_correction=False):
    I, S_klm = synthesize_spectrum(v,
                                   np.array([v0]),
                                   np.array([np.log(wG)]),
                                   np.array([np.log(wL)]),
                                   np.array([1.0]),
                                   folding_thresh=folding_thresh,
                                   log_correction=log_correction,
                                   )
    return I
