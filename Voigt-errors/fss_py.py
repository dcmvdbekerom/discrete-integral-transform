import numpy as np
import voigtlib

W = voigtlib.WeightOptimizer()

## Prepare width-axes based on number of gridpoints and linewidth data:
def init_w_axis(log_p, log_wi):
    log_w_min = np.min(log_wi)
    log_w_max = np.max(log_wi) + 1e-4
    N = np.ceil((log_w_max - log_w_min)/log_p) + 1
    log_w_arr = log_w_min + log_p * np.arange(N)
    return log_w_arr


## Calculate grid indices and grid alignments:   
def get_indices(arr_i, axis):
    pos    = np.interp(arr_i, axis, np.arange(axis.size))
    index  = pos.astype(int) 
    return index, index + 1, pos - index 


## Calculate Spectral matrix:
def calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i, awG_kind, awL_kind):
    
    #  Initialize matrix:
    S_klm = np.zeros((2 * v.size, log_wG.size, log_wL.size))

    #  Calculate grid indices and alignment:    
    ki0, ki1, tvi = get_indices(v0i, v)
    li0, li1, tau_wGi = get_indices(log_wGi, log_wG)
    mi0, mi1, tau_wLi = get_indices(log_wLi, log_wL)
    ni0, ni1, twLi = get_indices(np.exp(log_wLi), np.exp(log_wL))
    ni0, ni1, twGi = get_indices(np.exp(log_wGi), np.exp(log_wG))

    log_pwL = (log_wL[-1]-log_wL[0])/(len(log_wL)-1)
    log_pwG = (log_wG[-1]-log_wG[0])/(len(log_wG)-1)

    # Calculate weights:
    avi  = tvi

    W.calc_from_log_wGwL(log_wGi,log_wLi)
    aG0,aL0 = W.weights(tau_wGi,tau_wLi,log_pwG,log_pwL)

    if awG_kind == "linear":
        awGi = tau_wGi
    elif awG_kind == "opti":
        awGi = aG0
    elif awG_kind == "min-RMS":
        awGi = 0.5*tau_wGi + 0.5*(1-np.exp(-log_pwG*tau_wGi))/(1-np.exp(-log_pwG))
    elif awG_kind == "ZEP2":
        awGi = twGi * np.exp(log_wG[li1]-log_wGi)
    else:
        assert awG_kind == "ZEP"
        awGi = (1-np.exp(-log_pwG*tau_wGi))/(1-np.exp(-log_pwG))

    if awL_kind == "linear":
        awLi = tau_wLi
    elif awL_kind == "opti":
        awLi = aL0
    elif awL_kind == "min-RMS":
        awLi = 0.5*tau_wLi + 0.5*(1-np.exp(-log_pwL*tau_wLi))/(1-np.exp(-log_pwL))
    elif awL_kind == "ZEP2":
        awLi = twLi * np.exp(log_wL[mi1]-log_wLi)
    else:
        assert awL_kind == "ZEP"
        awLi = (1-np.exp(-log_pwL*tau_wLi))/(1-np.exp(-log_pwL))
        
    # Add lines to spectral matrix:
    np.add.at(S_klm, (ki0, li0, mi0), S0i * (1-avi) * (1-awGi) * (1-awLi))
    np.add.at(S_klm, (ki0, li0, mi1), S0i * (1-avi) * (1-awGi) *    awLi )
    np.add.at(S_klm, (ki0, li1, mi0), S0i * (1-avi) *    awGi  * (1-awLi))
    np.add.at(S_klm, (ki0, li1, mi1), S0i * (1-avi) *    awGi  *    awLi )
    np.add.at(S_klm, (ki1, li0, mi0), S0i *    avi  * (1-awGi) * (1-awLi))
    np.add.at(S_klm, (ki1, li0, mi1), S0i *    avi  * (1-awGi) *    awLi )
    np.add.at(S_klm, (ki1, li1, mi0), S0i *    avi  *    awGi  * (1-awLi))
    np.add.at(S_klm, (ki1, li1, mi1), S0i *    avi  *    awGi  *    awLi )
    
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
            gL_FT = np.exp(-np.pi*x*wL)
            gV_FT = gG_FT * gL_FT  / dv
            S_k_FT += np.fft.rfft(S_klm[:,l,m]) * gV_FT
    
    return np.fft.irfft(S_k_FT)[:v.size] 


## Synthesize spectrum:
def spectrum(v,
             log_pwG,
             log_pwL,
             v0i,
             log_wGi,
             log_wLi,
             S0i,
             awL_kind = 'ZEP',
             awG_kind = 'ZEP',
             ):

    # Only process lines within range:
    idx = (v0i >= np.min(v)) & (v0i < np.max(v))
    v0i, log_wGi, log_wLi, S0i = v0i[idx], log_wGi[idx], log_wLi[idx], S0i[idx]

    # Initialize width-axes:
    log_wG = init_w_axis(log_pwG,log_wGi)
    log_wL = init_w_axis(log_pwL,log_wLi)

    # Calculate spectral matrix & apply transform:
    S_klm = calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i,awG_kind, awL_kind)
    I = apply_transform(v, log_wG, log_wL, S_klm)
        
    return I,S_klm
