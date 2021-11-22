import numpy as np


gL_FT = lambda x,w: np.exp(-np.abs(x)*np.pi*w)
gG_FT = lambda x,w: np.exp(-(x*np.pi*w)**2/(4*np.log(2)))
gV_FT = lambda x,wG,wL: gG_FT(x,wG) * gL_FT(x,wL)


## Prepare width-axes based on number of gridpoints and linewidth data:
def init_w_axis(dx, log_wi, epsilon=1e-4):
    log_w_min = np.min(log_wi)
    log_w_max = np.max(log_wi)
    
    # Slightly increase max to fit lines that fall exactly on the boundary:
    log_w_max += epsilon

    N = np.ceil((log_w_max - log_w_min) / dx) + 1
    log_w_ax = log_w_min + dx * np.arange(N)
    return log_w_ax


## Calculate grid indices and grid alignments:   
def get_indices(arr_i, ax):
    pos    = np.interp(arr_i, ax, np.arange(ax.size))
    index  = pos.astype(int) 
    return index, index + 1, pos - index 


## Calculate lineshape distribution matrix:
def calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i):
    
    # Init the matrix with double the spectral size to prevent circular 
    # convolution:
    S_klm = np.zeros((2 * v.size, log_wG.size, log_wL.size))
    
    # Calculate grid indices and simple weights for every spectral line:
    ki0, ki1, avi = get_indices(v0i, v)          #Eqs 3.4 & 3.6
    li0, li1, aGi = get_indices(log_wGi, log_wG) #Eqs 3.7 & 3.10
    mi0, mi1, aLi = get_indices(log_wLi, log_wL) #Eqs 3.7 & 3.10
    
    # Distribute lines over matrix:
    np.add.at(S_klm, (ki0, li0, mi0), S0i * (1-avi) * (1-aGi) * (1-aLi))
    np.add.at(S_klm, (ki0, li0, mi1), S0i * (1-avi) * (1-aGi) *    aLi )
    np.add.at(S_klm, (ki0, li1, mi0), S0i * (1-avi) *    aGi  * (1-aLi))
    np.add.at(S_klm, (ki0, li1, mi1), S0i * (1-avi) *    aGi  *    aLi )
    np.add.at(S_klm, (ki1, li0, mi0), S0i *    avi  * (1-aGi) * (1-aLi))
    np.add.at(S_klm, (ki1, li0, mi1), S0i *    avi  * (1-aGi) *    aLi )
    np.add.at(S_klm, (ki1, li1, mi0), S0i *    avi  *    aGi  * (1-aLi))
    np.add.at(S_klm, (ki1, li1, mi1), S0i *    avi  *    aGi  *    aLi )
    
    return S_klm


def calc_gV_FT(x, wG, wL, dv, folding_thresh=1e-6):

    # Uncorrected result:
    result = gV_FT(x,wG,wL)
    
    # If the gV_FT doesn't go to zero at the Nyquist frequency, ringing is 
    # introduced. This can be corrected by "folding" the non-zero parts 
    # into the sampled frequency domain. Keep doing this until the gV_FT gets
    # sufficiently close to zero (below ```folding_thresh```)
    
    n = 1
    while gV_FT(n/(2*dv),wG,wL) >= folding_thresh:
        result += gV_FT(x + n/(2*dv), wG, wL)[::1-2*(n&1)]
        n += 1

    return result


## Apply transform:
def apply_transform(v,log_wG,log_wL,S_klm,folding_thresh):

    dv = (v[-1] - v[0]) / (v.size - 1)
    x  = np.fft.rfftfreq(2 * v.size, dv)
    nonzero = np.any(S_klm, axis=0)

    # Sum over lineshape distribution matrix -- Eqs 3.2 & 3.3:
    S_k_FT = np.zeros(v.size + 1, dtype=complex)
    for l in range(log_wG.size):
        for m in range(log_wL.size):
            if nonzero[l,m]:
                wG_l,wL_m = np.exp(log_wG[l]), np.exp(log_wL[m])
                gV_FT = calc_gV_FT(x, wG_l, wL_m, dv, folding_thresh)               
                S_k_FT += np.fft.rfft(S_klm[:,l,m]) * gV_FT
    
    return np.fft.irfft(S_k_FT)[:v.size] / dv


## Synthesize spectrum:
def synthesize_spectrum(v_ax,
                        v0, wG, wL, S0,
                        dxG = 0.1, dxL = 0.1,
                        folding_thresh = 1e-6):
    
    """Produce a broadened spectrum from a list of line positions, G&L-widths, and intensities.

        Parameters
        ----------
            v : array
                Array containing the spectral axis in cm-1
            v0 : array_like
                Array of line positions in cm-1
            wG : array_like
                Array of Gaussian widths in cm-1
            wL : array_like
                Array of Lorentizan widhts in cm-1
            S0 : array_like:
                Array of intensities
            dxG : float, optional
                Relative step size of the Gaussian width-grid (default=0.1)
            dxL : float, optional
                Relative step size of the Lorentzian width-grid (default=0.1) 
            folding_thresh : float, optional
                Threshold below which lineshape is considered zero for folding
                correction. (default=1e-6)


        Returns
        -------
        I : np.ndarray
            Broadened spectrum in units of [S0]/cm-1
        S_klm : np.ndarray
            Lineshape distribution matrix, indices are (v,wG,wL) in units of [S0]
        """

    # Select only lines within spectral range:
    idx = (v0 >= np.min(v_ax)) & (v0 < np.max(v_ax))
    v0, log_wG, log_wL, S0 = v0[idx], np.log(wG[idx]), np.log(wL[idx]), S0[idx]

    # Initialize width-axes:
    log_wG_ax = init_w_axis(dxG,log_wG) #Eq 3.8
    log_wL_ax = init_w_axis(dxL,log_wL) #Eq 3.9

    # Calculate matrix:
    S_klm = calc_matrix(v_ax, log_wG_ax, log_wL_ax, v0, log_wG, log_wL, S0)
    
    # Apply transform:
    I = apply_transform(v_ax, log_wG_ax, log_wL_ax, S_klm, folding_thresh)

    return I, S_klm


