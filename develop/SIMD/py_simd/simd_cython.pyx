#cython: language_level=3

import cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    float expf(float x)


cimport simd #imports the pxd file
cpdef add_flt(a, b):
    return simd.add_flt(a,b)
    
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)    
def cpp_add_at32(np.ndarray[np.float32_t, ndim=3] S_klm,
              np.ndarray[np.int32_t, ndim=1] k_arr,
              np.ndarray[np.int32_t, ndim=1] l_arr,
              np.ndarray[np.int32_t, ndim=1] m_arr,
              np.ndarray[np.float32_t, ndim=1] values):

    simd.cpp_add_at32(
        <float*> &S_klm[0,0,0], 
        <int*> &k_arr[0], 
        <int*> &l_arr[0], 
        <int*> &m_arr[0], 
        <float*> &values[0], 
        <int> S_klm.shape[0], 
        <int> S_klm.shape[1], 
        <int> S_klm.shape[2], 
        <int> values.shape[0])
        
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


def cpp_calc_matrix(np.ndarray[np.float32_t, ndim=2] database,
                        np.ndarray[np.float32_t, ndim=3] S_klm,
                        float v_min, float log_wG_min, float log_wL_min, 
                        float dv, float dxG, float dxL):

    simd.cpp_calc_matrix(
        <float*> &database[0,0], 
        <float*> &S_klm[0,0,0], 
        <float> v_min,
        <float> log_wG_min,
        <float> log_wL_min,
        <float> dv,
        <float> dxG,
        <float> dxL,   
        <int> S_klm.shape[0], 
        <int> S_klm.shape[1], 
        <int> S_klm.shape[2], 
        <int> database.shape[0])