import numpy as np

## Define constants for optimized weights:
C1_GG = ((6 * np.pi - 16) / (15 * np.pi - 32)) ** (1 / 1.50)
C1_LG = ((6 * np.pi - 16) / 3 * (np.log(2) / (2 * np.pi)) ** 0.5) ** (1 / 2.25)
C2_GG = (2 * np.log(2) / 15) ** (1 / 1.50)
C2_LG = ((2 * np.log(2)) ** 2 / 15) ** (1 / 2.25)


## Prepare width-axes based on number of gridpoints and linewidth data:
def init_w_axis(dx, log_wi):
    log_w_min = np.min(log_wi)
    log_w_max = np.max(log_wi) + 1e-4
    N = np.ceil((log_w_max - log_w_min)/dx) + 1
    log_w_arr = log_w_min + dx * np.arange(N)
    return log_w_arr


## Calculate grid indices and grid alignments:   
def get_indices(arr_i, axis):
    pos    = np.interp(arr_i, axis, np.arange(axis.size))
    index  = pos.astype(int) 
    return index, index + 1, pos - index 


## Calculate lineshape distribution matrix:
def calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i, aw_kind):
    
    #  Initialize matrix:
    S_klm = np.zeros((2 * v.size, log_wG.size, log_wL.size))

    #  Calculate grid indices and alignment:    
    ki0, ki1, tvi = get_indices(v0i, v)
    li0, li1, tGi = get_indices(log_wGi, log_wG)
    mi0, mi1, tLi = get_indices(log_wLi, log_wL)

    # Calculate weights:
    if aw_kind == "optimized":

        dv = (v[-1] - v[0])/(v.size - 1)
        dxvGi = dv / np.exp(log_wGi)
        dxG = (log_wG[-1] - log_wG[0])/(log_wG.size - 1)
        dxL = (log_wL[-1] - log_wL[0])/(log_wL.size - 1)

        alpha_i = np.exp(log_wLi - log_wGi)

        R_Gv = 8 * np.log(2)
        R_GG = 2 - 1 / (C1_GG + C2_GG * alpha_i ** (2 / 1.50)) ** 1.50
        R_GL = -2 * np.log(2) * alpha_i ** 2

        R_LL = 1
        R_LG = 1 / (C1_LG * alpha_i ** (1 / 2.25) + C2_LG * alpha_i ** (4 / 2.25)) ** 2.25
        
        avi = tvi

        aGi = tGi + (R_Gv * tvi * (tvi - 1) * dxvGi**2 +
                     R_GG * tGi * (tGi - 1) * dxG**2 +
                     R_GL * tLi * (tLi - 1) * dxL**2 ) / (2 * dxG)
        
        aLi = tLi + (R_LG * tGi * (tGi - 1) * dxG**2 +
                     R_LL * tLi * (tLi - 1) * dxL**2 ) / (2 * dxL)

    else:
        avi = tvi
        aGi = tGi
        aLi = tLi
    
    # Add lines to spectral matrix:
    np.add.at(S_klm, (ki0, li0, mi0), S0i * (1-avi) * (1-aGi) * (1-aLi))
    np.add.at(S_klm, (ki0, li0, mi1), S0i * (1-avi) * (1-aGi) *    aLi )
    np.add.at(S_klm, (ki0, li1, mi0), S0i * (1-avi) *    aGi  * (1-aLi))
    np.add.at(S_klm, (ki0, li1, mi1), S0i * (1-avi) *    aGi  *    aLi )
    np.add.at(S_klm, (ki1, li0, mi0), S0i *    avi  * (1-aGi) * (1-aLi))
    np.add.at(S_klm, (ki1, li0, mi1), S0i *    avi  * (1-aGi) *    aLi )
    np.add.at(S_klm, (ki1, li1, mi0), S0i *    avi  *    aGi  * (1-aLi))
    np.add.at(S_klm, (ki1, li1, mi1), S0i *    avi  *    aGi  *    aLi )
    
    return S_klm


## Apply transform:
def apply_transform(v,log_wG,log_wL,S_klm):

    dv    = (v[-1] - v[0]) / (v.size - 1)
    x     = np.arange(v.size + 1) / (2 * v.size * dv)

    # Sum over spectral matrix:
    S_k_FT = np.zeros(v.size + 1, dtype = np.complex64)
    for l in range(log_wG.size):
        for m in range(log_wL.size):
            wG,wL = np.exp(log_wG[l]),np.exp(log_wL[m])
            gG_FT = np.exp(-(np.pi*x*wG)**2/(4*np.log(2)))
            gL_FT = np.exp(- np.pi*x*wL)
            gV_FT = gG_FT * gL_FT
            S_k_FT += np.fft.rfft(S_klm[:,l,m]) * gV_FT
    
    return np.fft.irfft(S_k_FT)[:v.size] / dv


## Synthesize spectrum:
def synthesize_spectrum(v, v0i, log_wGi, log_wLi, S0i, dxG = 0.14, dxL = 0.2, aw_kind = 'simple'):

    # Only process lines within range:
    idx = (v0i >= np.min(v)) & (v0i < np.max(v))
    v0i, log_wGi, log_wLi, S0i = v0i[idx], log_wGi[idx], log_wLi[idx], S0i[idx]

    # Initialize width-axes:
    log_wG = init_w_axis(dxG,log_wGi)
    log_wL = init_w_axis(dxL,log_wLi)

    # Calculate spectral matrix & apply transform:
    S_klm = calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i, aw_kind)
    I = apply_transform(v, log_wG, log_wL, S_klm)
        
    return I,S_klm
