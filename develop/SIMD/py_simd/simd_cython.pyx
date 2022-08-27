#cython: language_level=3

import cython
import numpy as np
cimport numpy as np

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
        
        
def cpp_calc_matrix_avx(np.ndarray[np.float32_t, ndim=2] database,
                        np.ndarray[np.float32_t, ndim=3] S_klm,
                        float v_min, float log_wG_min, float log_wL_min, 
                        float dv, float dxG, float dxL):

    simd.cpp_calc_matrix_avx(
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