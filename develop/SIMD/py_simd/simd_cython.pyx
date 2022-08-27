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
def add_at32(np.ndarray[np.float32_t, ndim=3] arr,
              np.ndarray[np.int32_t, ndim=1] k_arr,
              np.ndarray[np.int32_t, ndim=1] l_arr,
              np.ndarray[np.int32_t, ndim=1] m_arr,
              np.ndarray[np.float32_t, ndim=1] values):

    cdef unsigned int N = values.size
    cdef unsigned int i

    for i in range(N):
        arr[k_arr[i],l_arr[i],m_arr[i]] += values[i]

    return arr