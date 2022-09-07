#cython: language_level=3

import cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    float expf(float x)
    double exp(double x)

from scipy.special import lambertw
from scipy.fft import next_fast_len
from scipy.signal._signaltools import _calc_oa_lens



def convolve_and_add_to_output():
    return

def add_circ_buffer():
    return

def add_lines_and_apply_lineshape(
    S_klm,
    np.ndarray[np.float32_t, ndim=1] S0_arr,
    np.ndarray[np.float32_t, ndim=1] v0_arr,
    np.ndarray[np.float32_t, ndim=1] v1_arr,
    np.ndarray[np.float32_t, ndim=1] wG_arr,
    np.ndarray[np.float32_t, ndim=1] wL_arr,
    float v_min, float log_wG_min, float log_wL_min, 
    float dv, float dxG, float dxL
    np.ndarray[np.float32_t, ndim=1] log_wG, 
    np.ndarray[np.float32_t, ndim=1] log_wL,
    truncation = 50.0, #cm-1
    wstep = 0.01, #cm-1
    ):
    
    Nv, NwG, NwL #are presumed known
    
    lineshape_size = 2*int(truncation/wstep) #left-to-right
    
    (block_size, 
    overlap_size, 
    in1_step, 
    in2_step) = _calc_oa_lens(Nv, lineshape_size)

    
    v_diff_min, v_diff_max = p * da_min / dv, p * da_max / dv
    k_diff_min, k_diff_max = np.floor(v_diff_min), np.ceil(k_diff_min)
    v_min2 = v_min + v_diff_min
    
    spread_size = k_diff_max - k_diff_min + 1 # +1 to accomodate k0 + 1
    circ_buffer_size = 2**np.ceil(np.log2(spread_len))
    circ_mask = circ_buffer_size - 1 #if spread len is 0b01000, circ_mask is 0b00111
    
    # S_circ is a circular buffer that temporarily stores results.
    # every time iv increments, the circular buffer increments by 1.
    # The size is the smallest power-of-2 that fits spread_len, so
    # that wrapping the pointer can be done cheaply.
    S_circ = np.zeros((circ_buffer_size,NwG,NwL))
    S_klm = np.zeros((block_size,NwG,NwL))
    
    np.ndarray[np.float32_t, ndim=3]):
    
    cdef float k,l,m, av,awG,awL
    cdef int k0,k1,l0,l1,m0,m1
    cdef float S0i, v0i, wGi, wLi
    cdef int Nlines = <int> S0_arr.shape[0]
    cdef int il, iv
    old_iv = -1
    
    running_index_arr = np.zeros((NwG,NwL))
    distance_arr = np.zeros((NwG,NwL))
    cum_skip_arr = np.zeros((NwG,NwL))
    last_updated_arr = np.zeros((NwG,NwL))
    
    skip_index_start = np.zeros((NwG, NwL), dtype=list)
    skip_index_stop = np.zeros((NwG, NwL), dtype=list)
    
    ib = 0 # index of the v-axis in the block array
    io = 0 # index of the v-axis in the output array
    
    for il in len(v0_arr):
        
        ## iv is the integer value for v0, so before pressure shift. 
        ## importantly, iv is monotonically increasing (k is not!)
        v0i = v0_arr[il]
        iv = (v0i - v_min - v_diff_min) / dv 
        if iv != old_iv and iv >= 0:
        
            ## Values that are guaranteed not to change anymore
            ## are copied from circular buffer to block array
            ## We do this as often as needed to reset circular pointer to current position of the block pointer.
            ## After `spread_size` additions the circular buffer should be all zeros.
            ## (so no need to continue till spread_size2 or even iv_step)
            
            k = old_iv - k_diff_min
            iv_step = iv - iv_old
            for j in range(np.min(iv_step, spread_size)):
                
                #Add circular buffer to block array:
                for l in range(NwG):
                    for m in range(NwL):
                        val_klm = S_circ[(k + j) & circ_mask, l, m] 
                        if val_klm != 0.0:
                            if ib - last_updated_arr[l,m] > linehsape_size:
                                ## A new jump is needed:
                                cum_skip_arr[l,m] += ib - last_updated_arr[l,m] - lineshape_size
                                # TO-DO: update skip_index_arr
                                                      
                            ib_adj = ib - cum_skip_arr[l,m]
                            
                            if ib_adj > block_size:
                                # Convolve
                                # Multiply with lineshape function
                                # Add to output array
                                # Clear block
                            
                            S_klm[ib_adj] = val_klm
                            S_circ[(k + j) & circ_mask, l, m] = 0.0
                            last_updated_arr[l,m] = ib

                ib += 1
            old_iv = iv
        
        #continue adding lines to circular buffer
        S0i = S0_arr[il]
        v1i = v0_arr[il] + p * da_arr[il]
        wGi = wG_arr[il]
        wLi = wL_arr[il]
        
        k = (v1i - v_min) / dv
        k0 = <int>k
        k1 = k0 + 1
        av = k - k0

        l = (wGi - log_wG_min) / dxG
        l0 = <int>l
        l1 = l0 + 1
        awG = l - l0

        m = (wLi - log_wL_min) / dxL
        m0 = <int>m
        m1 = m0 + 1
        awL = m - m0
        
        k0 &= circ_mask
        k1 &= circ_mask

        S_circ[k0, l0, m0] += (1-av) * (1-awG) * (1-awL) * S0i
        S_circ[k0, l0, m1] += (1-av) * (1-awG) *    awL  * S0i
        S_circ[k0, l1, m0] += (1-av) *    awG  * (1-awL) * S0i
        S_circ[k0, l1, m1] += (1-av) *    awG  *    awL  * S0i
        S_circ[k1, l0, m0] +=    av  * (1-awG) * (1-awL) * S0i
        S_circ[k1, l0, m1] +=    av  * (1-awG) *    awL  * S0i
        S_circ[k1, l1, m0] +=    av  *    awG  * (1-awL) * S0i
        S_circ[k1, l1, m1] +=    av  *    awG  *    awL  * S0i
























    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def cy_add_at32(np.ndarray[np.float32_t, ndim=3] arr,
              np.ndarray[np.int32_t, ndim=1] k_arr,
              np.ndarray[np.int32_t, ndim=1] l_arr,
              np.ndarray[np.int32_t, ndim=1] m_arr,
              np.ndarray[np.float32_t, ndim=1] values):

    cdef unsigned int N = values.size
    cdef unsigned int i

    for i in range(N):
        arr[k_arr[i],l_arr[i],m_arr[i]] += values[i]

    return arr
    
cpdef add_flt(float a, float b):
    return a + b

        
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False) 
def cy_multiply_lineshape(np.ndarray[np.complex64_t, ndim=3] S_klm_FT, 
                            float dv,
                            np.ndarray[np.float32_t, ndim=1] log_wG, 
                            np.ndarray[np.float32_t, ndim=1] log_wL,
                            ):

    cdef int Nx  = <int>S_klm_FT.shape[0]
    cdef int NwG = <int>S_klm_FT.shape[1]
    cdef int NwL = <int>S_klm_FT.shape[2]

    cdef float pi = 3.141592653589793
    cdef float log2 = 0.6931471805599453
    cdef float x_step = 1 / (2 * dv * (Nx - 1))

    cdef np.ndarray[np.complex64_t, ndim=1] S_k_FT = np.zeros(Nx, dtype=np.complex64)
    
    cdef float x_k, wG_l, wL_m, gV_FT
    cdef int k,l,m
  
    for l in range(NwG):
        wG_l = expf(log_wG[l])
        for m in range(NwL):
            wL_m = expf(log_wL[m])
            for k in range(Nx):
                x_k = k * x_step
                gV_FT = expf(-((pi*wG_l*x_k)**2/(4*log2) + pi*wL_m*x_k))
                S_k_FT[k] = S_k_FT[k] + S_klm_FT[k,l,m] * gV_FT        
    
    return S_k_FT / dv
        

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)    
def cy_calc_matrix( np.ndarray[np.float32_t, ndim=3] S_klm,
                    np.ndarray[np.float32_t, ndim=1] S0_arr,
                    np.ndarray[np.float32_t, ndim=1] v0_arr,
                    np.ndarray[np.float32_t, ndim=1] wG_arr,
                    np.ndarray[np.float32_t, ndim=1] wL_arr,
                    float v_min, float log_wG_min, float log_wL_min, 
                    float dv, float dxG, float dxL):
    
    cdef float k,l,m, av,awG,awL
    cdef int k0,k1,l0,l1,m0,m1
    cdef float S0i, v0i, wGi, wLi
    cdef int Nlines = <int> S0_arr.shape[0]
    
    for i in range(Nlines):
    
        S0i = S0_arr[i]
        v0i = v0_arr[i]
        wGi = wG_arr[i]
        wLi = wL_arr[i]
  
        k = (v0i - v_min) / dv
        k0 = <int>k
        k1 = k0 + 1
        av = k - k0

        l = (wGi - log_wG_min) / dxG
        l0 = <int>l
        l1 = l0 + 1
        awG = l - l0

        m = (wLi - log_wL_min) / dxL
        m0 = <int>m
        m1 = m0 + 1
        awL = m - m0

        S_klm[k0, l0, m0] += (1-av) * (1-awG) * (1-awL) * S0i
        S_klm[k0, l0, m1] += (1-av) * (1-awG) *    awL  * S0i
        S_klm[k0, l1, m0] += (1-av) *    awG  * (1-awL) * S0i
        S_klm[k0, l1, m1] += (1-av) *    awG  *    awL  * S0i
        S_klm[k1, l0, m0] +=    av  * (1-awG) * (1-awL) * S0i
        S_klm[k1, l0, m1] +=    av  * (1-awG) *    awL  * S0i
        S_klm[k1, l1, m0] +=    av  *    awG  * (1-awL) * S0i
        S_klm[k1, l1, m1] +=    av  *    awG  *    awL  * S0i


