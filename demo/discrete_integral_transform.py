import numpy as np

## Define constants for optimized weights (Eq 3.20 & Table 1):
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
def calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i, optimized):
    
    #  Initialize matrix:
    S_klm = np.zeros((2 * v.size, log_wG.size, log_wL.size))

    #  Calculate grid indices and alignment:    
    ki0, ki1, tvi = get_indices(v0i, v)          #Eqs 3.4 & 3.6
    li0, li1, tGi = get_indices(log_wGi, log_wG) #Eqs 3.7 & 3.10
    mi0, mi1, tLi = get_indices(log_wLi, log_wL) #Eqs 3.7 & 3.10

    # Calculate weights:
    if optimized:
        #Eqs 3.18 & 3.19:

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
        #Eq 3.11:
        avi = tvi
        aGi = tGi
        aLi = tLi

    indices = [ (ki0, li0, mi0),
                (ki0, li0, mi1),
                (ki0, li1, mi0),
                (ki0, li1, mi1),
                (ki1, li0, mi0),
                (ki1, li0, mi1),
                (ki1, li1, mi0),
                (ki1, li1, mi1)]

    intensities = [S0i * (1-avi) * (1-aGi) * (1-aLi)),
                   S0i * (1-avi) * (1-aGi) *    aLi ),
                   S0i * (1-avi) *    aGi  * (1-aLi)),
                   S0i * (1-avi) *    aGi  *    aLi ),
                   S0i *    avi  * (1-aGi) * (1-aLi)),
                   S0i *    avi  * (1-aGi) *    aLi ),
                   S0i *    avi  *    aGi  * (1-aLi)),
                   S0i *    avi  *    aGi  *    aLi )]

    # Add lines to matrix -- Eqs 2.8 & 2.10:
    for klm, I in zip(indices, intensities):
        np.add.at(S_klm, klm, I)
    
    return S_klm, indices, intensities


def calc_gV_FT(x,wG,wL,folding_thresh):

    gV_FT = lambda x,wG,wL: np.exp(-((np.pi*x*wG)**2/(4*np.log(2)) + np.pi*x*wL))
    x_fold = (x,x[::-1])
    
    result = gV_FT(x,wG,wL)
    n = 1
    while gV_FT(n/(2*dv),wG,wL) >= folding_thresh:
        result += gV_FT(n/(2*dv) + x_fold[n&1],wG,wL)
        n += 1

    return result


## Apply transform:
def apply_transform(v,log_wG,log_wL,S_klm,folding_thresh):

    dv     = (v[-1] - v[0]) / (v.size - 1)
    x      = np.fft.rfftfreq(2 * v.size, dv)

    # Sum over lineshape distribution matrix -- Eqs 3.2 & 3.3:
    S_k_FT = np.zeros(v.size + 1, dtype = np.complex64)
    for l in range(log_wG.size):
        for m in range(log_wL.size):
            wG_l,wL_m = np.exp(log_wG[l]),np.exp(log_wL[m])
            gV_FT = calc_gV_FT(x,wG_l,wL_m,folding_thresh)               
            S_k_FT += np.fft.rfft(S_klm[:,l,m]) * gV_FT
    
    return np.fft.irfft(S_k_FT)[:v.size] / dv


def jacobian_integrals(v, log_wG, log_wL, Isyn, Iexp, folding_thresh = 1e-6):

    dv  = (v[-1] - v[0]) / (v.size - 1)
    dxL = (log_wL[-1] - log_wL[0])/(log_wL.size - 1)
    fL  =  np.exp(0.5 * dxL)
    wG  = np.exp(log_wG)
    wL  = np.exp(log_wL)

    ##Note: we can't use the I_FT from earlier because the latter half
    ## must be zeroed first to prevent circular convolution.
    ## This adds one FT per step, which could be eliminated if
    ## circular convolution is not an issue.
    
    DeltaI = np.zeros(2 * v.size)
    DeltaI[:v.size] = Iexp - Isyn
    DeltaI_FT = np.fft.rfft(DeltaI) * dv

    int1 = np.sum(DeltaI * Iexp) * dv
    int2 = np.sum(DeltaI**2) * dv

    IS0 = np.zeros((2 * v.size, log_wG.size, log_wL.size))
    Iv0 = np.zeros((2 * v.size, log_wG.size, log_wL.size))
    IwG = np.zeros((2 * v.size, log_wG.size, log_wL.size))
    IwL = np.zeros((2 * v.size, log_wG.size, log_wL.size))

    for l in range(wG.size):
        gV_FT = calc_gV_FT(x,wG[l], wL[0] / fL, folding_thresh)
        Iconv_prev = np.fft.irfft(DeltaI_FT * gV_FT)

        for m in range(wL.size):
            gV_FT = calc_gV_FT(x, wG[l], wL[m] * fL, folding_thresh)
            Iconv_next = np.fft.irfft(DeltaI_FT * gV_FT)

            IS0[ :  ,l,m] = (Iconv_prev + Iconv_next) / 2
            
            Iv0[1:-1,l,m] = (IS0[2:,l,m] - IS0[:-2,l,m]) / (2*dv)
            Iv0[0   ,l,m] = (IS0[1 ,l,m] - IS0[ -1,l,m]) / (2*dv)

            IwG[1:-1,l,m] = (IS0[2:,l,m] + IS0[:-2,l,m] - 2 * IS0[1:-1,l,m]) / (8 * np.log(2) * dv**2)
            IwG[0   ,l,m] = (IS0[1 ,l,m] + IS0[ -1,l,m] - 2 * IS0[0   ,l,m]) / (8 * np.log(2) * dv**2)

            IwL[ :  ,l,m] = (Iconv_next - Iconv_prev) / (wL[m] * dxL)

            Iconv_prev = Iconv_next            

    return int1, int2, IS0, Iv0, IwG, IwL


def calc_jacobian(integrals, indices, intensities, S0i, wGi):

    int1,int2,IS0,Iv0,IwG,IwL = integrals

    int_DS0 = np.zeros(S0i.size)
    int_Dv0 = np.zeros(S0i.size)
    int_DwG = np.zeros(S0i.size)
    int_DwL = np.zeros(S0i.size)

    for klm, I in zip(indices, intensities):
        int_DS0 += I * IS0[klm]
        int_Dv0 += I * Iv0[klm]
        int_DwG += I * IwG[klm]
        int_DwL += I * IwL[klm]

    int_DS0 /= Si0
    int_DwG *= wGi
    
    dLdS0 = 2 * (int1 - int_DS0) / int2
    dLdv0 = 2 * (int1 - int_Dv0) / int2
    dLdwG = 2 * (int1 - int_DwG) / int2
    dLdwL = 2 * (int1 - int_DwL) / int2
    
    return dLdS0, dLdv0, dLdwG, dLdwL
    

## Synthesize spectrum:
def synthesize_spectrum(v,
                        v0i, log_wGi, log_wLi, S0i,
                        dxG = 0.14, dxL = 0.2,
                        Iexp = None,
                        optimized = False, folding_thresh = 1e-6):

    # Only process lines within range:
    idx = (v0i >= np.min(v)) & (v0i < np.max(v))
    v0i, log_wGi, log_wLi, S0i = v0i[idx], log_wGi[idx], log_wLi[idx], S0i[idx]

    # Initialize width-axes:
    log_wG = init_w_axis(dxG,log_wGi) #Eq 3.8
    log_wL = init_w_axis(dxL,log_wLi) #Eq 3.9

    # Calculate matrix & apply transform:
    S_klm, indices, intensities = calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i, optimized)
    I = apply_transform(v, log_wG, log_wL, S_klm, folding_thresh)

    # Calculate Jacobian:
    if Iexp == None:
        J = tuple(4*[np.zeros(Si0.size)])
    else:
        integrals = jacobian_integrals(v, log_wG, log_wL, I, Iexp)
        J = calc_jacobian(integrals, indices, intensities, S0i, np.exp(log_wGi))
        
    return I, S_klm, J


