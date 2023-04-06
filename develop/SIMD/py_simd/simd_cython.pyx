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
def cy0_calc_matrix_222(np.ndarray[np.float32_t, ndim=3] S_klm,
                    np.ndarray[np.float32_t, ndim=1] S0_arr,
                    np.ndarray[np.float32_t, ndim=1] v0_arr,
                    np.ndarray[np.float32_t, ndim=1] log_wG_arr,
                    np.ndarray[np.float32_t, ndim=1] log_wL_arr,
                    float v_min, float log_wG_min, float log_wL_min, 
                    float dv, float dxG, float dxL):
    
    cdef float k,l,m, av,awG,awL
    cdef int k0,k1,l0,l1,m0,m1,i
    cdef float S0i, v0i, log_wGi, log_wLi
    cdef int Nlines = <int> S0_arr.shape[0]
    
    for i in range(Nlines):
    
        S0i = S0_arr[i]
        v0i = v0_arr[i]
        log_wGi = log_wG_arr[i]
        log_wLi = log_wL_arr[i]
  
        k = (v0i - v_min) / dv
        k0 = <int>k
        k1 = k0 + 1
        av = k - k0

        l = (log_wGi - log_wG_min) / dxG
        l0 = <int>l
        l1 = l0 + 1
        awG = l - l0

        m = (log_wLi - log_wL_min) / dxL
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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)    
def cy0_calc_matrix_333(np.ndarray[np.float32_t, ndim=3] S_klm,
                          np.ndarray[np.float32_t, ndim=1] S0_arr,
                          np.ndarray[np.float32_t, ndim=1] v0_arr,
                          np.ndarray[np.float32_t, ndim=1] log_wG_arr,
                          np.ndarray[np.float32_t, ndim=1] log_wL_arr,
                          float v_min, float log_wG_min, float log_wL_min, 
                          float dv, float dxG, float dxL):
    
    cdef float k,l,m, tv,tG,tL
    cdef int k0,k1,k2,l0,l1,l2,m0,m1,m2,i
    cdef float S0i, v0i, log_wGi, log_wLi
    cdef int Nlines = <int> S0_arr.shape[0]
    
    for i in range(Nlines):
    
        S0i = S0_arr[i]
        v0i = v0_arr[i]
        log_wGi = log_wG_arr[i]
        log_wLi = log_wL_arr[i]
  
        k = (v0i - v_min) / dv
        k0 = <int>(k + 0.5)
        k1 = k0 + 1
        k2 = k0 + 2
        tv = k - k0

        l = (log_wGi - log_wG_min) / dxG
        l0 = <int>(l + 0.5)
        l1 = l0 + 1
        l2 = l0 + 2
        tG = l - l0

        m = (log_wLi - log_wL_min) / dxL
        m0 = <int>(m + 0.5)
        m1 = m0 + 1
        m2 = m0 + 2
        tL = m - m0

        
        S_klm[k0, l0, m0] += 0.5*tv*(tv-1) * 0.5*tG*(tG-1) * 0.5*tL*(tL-1) * S0i
        S_klm[k0, l0, m1] += 0.5*tv*(tv-1) * 0.5*tG*(tG-1) *     (1-tL**2) * S0i
        S_klm[k0, l0, m2] += 0.5*tv*(tv-1) * 0.5*tG*(tG-1) * 0.5*tL*(tL+1) * S0i

        S_klm[k0, l1, m0] += 0.5*tv*(tv-1) *     (1-tG**2) * 0.5*tL*(tL-1) * S0i
        S_klm[k0, l1, m1] += 0.5*tv*(tv-1) *     (1-tG**2) *     (1-tL**2) * S0i
        S_klm[k0, l1, m2] += 0.5*tv*(tv-1) *     (1-tG**2) * 0.5*tL*(tL+1) * S0i      
        
        S_klm[k0, l2, m0] += 0.5*tv*(tv-1) * 0.5*tG*(tG+1) * 0.5*tL*(tL-1) * S0i
        S_klm[k0, l2, m1] += 0.5*tv*(tv-1) * 0.5*tG*(tG+1) *     (1-tL**2) * S0i
        S_klm[k0, l2, m2] += 0.5*tv*(tv-1) * 0.5*tG*(tG+1) * 0.5*tL*(tL+1) * S0i


        S_klm[k1, l0, m0] +=     (1-tG**2) * 0.5*tG*(tG-1) * 0.5*tL*(tL-1) * S0i
        S_klm[k1, l0, m1] +=     (1-tG**2) * 0.5*tG*(tG-1) *     (1-tL**2) * S0i
        S_klm[k1, l0, m2] +=     (1-tG**2) * 0.5*tG*(tG-1) * 0.5*tL*(tL+1) * S0i

        S_klm[k1, l1, m0] +=     (1-tG**2) *     (1-tG**2) * 0.5*tL*(tL-1) * S0i
        S_klm[k1, l1, m1] +=     (1-tG**2) *     (1-tG**2) *     (1-tL**2) * S0i
        S_klm[k1, l1, m2] +=     (1-tG**2) *     (1-tG**2) * 0.5*tL*(tL+1) * S0i      
        
        S_klm[k1, l2, m0] +=     (1-tG**2) * 0.5*tG*(tG+1) * 0.5*tL*(tL-1) * S0i
        S_klm[k1, l2, m1] +=     (1-tG**2) * 0.5*tG*(tG+1) *     (1-tL**2) * S0i
        S_klm[k1, l2, m2] +=     (1-tG**2) * 0.5*tG*(tG+1) * 0.5*tL*(tL+1) * S0i        
        
        
        S_klm[k2, l0, m0] += 0.5*tv*(tv+1) * 0.5*tG*(tG-1) * 0.5*tL*(tL-1) * S0i
        S_klm[k2, l0, m1] += 0.5*tv*(tv+1) * 0.5*tG*(tG-1) *     (1-tL**2) * S0i
        S_klm[k2, l0, m2] += 0.5*tv*(tv+1) * 0.5*tG*(tG-1) * 0.5*tL*(tL+1) * S0i

        S_klm[k2, l1, m0] += 0.5*tv*(tv+1) *     (1-tG**2) * 0.5*tL*(tL-1) * S0i
        S_klm[k2, l1, m1] += 0.5*tv*(tv+1) *     (1-tG**2) *     (1-tL**2) * S0i
        S_klm[k2, l1, m2] += 0.5*tv*(tv+1) *     (1-tG**2) * 0.5*tL*(tL+1) * S0i      
        
        S_klm[k2, l2, m0] += 0.5*tv*(tv+1) * 0.5*tG*(tG+1) * 0.5*tL*(tL-1) * S0i
        S_klm[k2, l2, m1] += 0.5*tv*(tv+1) * 0.5*tG*(tG+1) *     (1-tL**2) * S0i
        S_klm[k2, l2, m2] += 0.5*tv*(tv+1) * 0.5*tG*(tG+1) * 0.5*tL*(tL+1) * S0i
        
        
        
        

# Same as cy0, but calculates pressure broadening inside cython function.
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)    
def cy1_calc_matrix( np.ndarray[np.float32_t, ndim=3] S_klm,
                     np.ndarray[np.float32_t, ndim=1] S0_arr,
                     np.ndarray[np.float32_t, ndim=1] v0_arr,
                     np.ndarray[np.float32_t, ndim=1] da_arr,
                     np.ndarray[np.float32_t, ndim=1] log_wG_arr,
                     np.ndarray[np.float32_t, ndim=1] log_wL_arr,
                     float p,
                     float v_min, float log_wG_min, float log_wL_min, 
                     float dv, float dxG, float dxL):
                     
    cdef float k_,l_,m_, av,awG,awL
    cdef int k0,k1,l0,l1,m0,m1,il
    cdef float S0i, vii, log_wGi, log_wLi
    cdef int Nlines = <int> S0_arr.shape[0]

    for il in range(Nlines):

        S0i = S0_arr[il]
        vii = v0_arr[il] + p * da_arr[il]
        log_wGi = log_wG_arr[il]
        log_wLi = log_wL_arr[il]
        
        k_ = (vii - v_min) / dv
        k0 = <int>k_
        k1 = k0 + 1
        av = k_ - k0

        l_ = (log_wGi - log_wG_min) / dxG
        l0 = <int>l_
        l1 = l0 + 1
        awG = l_ - l0

        m_ = (log_wLi - log_wL_min) / dxL
        m0 = <int>m_
        m1 = m0 + 1
        awL = m_ - m0

        if k0 >= 0:

            S_klm[k0, l0, m0] += (1-av) * (1-awG) * (1-awL) * S0i
            S_klm[k0, l0, m1] += (1-av) * (1-awG) *    awL  * S0i
            S_klm[k0, l1, m0] += (1-av) *    awG  * (1-awL) * S0i
            S_klm[k0, l1, m1] += (1-av) *    awG  *    awL  * S0i
            S_klm[k1, l0, m0] +=    av  * (1-awG) * (1-awL) * S0i
            S_klm[k1, l0, m1] +=    av  * (1-awG) *    awL  * S0i
            S_klm[k1, l1, m0] +=    av  *    awG  * (1-awL) * S0i
            S_klm[k1, l1, m1] +=    av  *    awG  *    awL  * S0i


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)    
def cy2_calc_matrix( np.ndarray[np.float32_t, ndim=3] S_klm,
                     np.ndarray[np.float32_t, ndim=1] S0_arr,
                     np.ndarray[np.float32_t, ndim=1] v0_arr,
                     np.ndarray[np.float32_t, ndim=1] da_arr,
                     np.ndarray[np.float32_t, ndim=1] log_wG_arr,
                     np.ndarray[np.float32_t, ndim=1] log_wL_arr,
                     float p,
                     float v_min, float log_wG_min, float log_wL_min, 
                     float dv, float dxG, float dxL,
                     int k_diff_min, int spread_size):
                     
    cdef float k_,l_,m_, av,awG,awL, val_klm
    cdef int k0,k1,l0,l1,m0,m1,il
    cdef float S0i, v0i, vii, log_wGi, log_wLi
    cdef int Nlines = <int> S0_arr.shape[0]
    cdef int iv, k, j, kj, j_max


    cdef int NwG = <int>S_klm.shape[1]
    cdef int NwL = <int>S_klm.shape[2]

    cdef int circ_buffer_size = int(2**np.ceil(np.log2(spread_size)))
    cdef int circ_mask = circ_buffer_size - 1 #if spread len is 0b01000, circ_mask is 0b00111
    cdef np.ndarray[np.float32_t, ndim=3] S_circ = np.zeros((circ_buffer_size, NwG, NwL), dtype=np.float32)

    cdef int iv_old = 0
    for il in range(Nlines):

        v0i = v0_arr[il]
        iv = int((v0_arr[il] - v_min) / dv) 
        if iv != iv_old:
            
            k = iv_old + k_diff_min
            iv_step = iv - iv_old
            j_max = min(iv_step, spread_size)
            #The index kj is the index of the v-grid.
            #Add the circular buffer to the block as much as needed
            #to catch up with the change in iv.
            for j in range(j_max):
                kj = k + j
                kj_c = kj & circ_mask

                #Iterate over the wG x wL grid; if value in the circular
                #buffer is non-zero add it to the block memory.
                #TO-DO: this should be compiled into a memcpy.
                
                S_klm[kj] = S_circ[kj_c]
                S_circ[kj_c] = 0.0
                
                #for l in range(NwG):
                #    for m in range(NwL):
                #        val_klm = S_circ[kj_c, l, m]
                #        if val_klm != 0.0:
                #            
                #            S_klm[kj, l, m] = val_klm 
                #            S_circ[kj_c, l, m] = 0.0

            iv_old = iv
            
        #Continue adding lines to circular buffer
        S0i = S0_arr[il]
        vii = v0_arr[il] + p * da_arr[il]
        log_wGi = log_wG_arr[il]
        log_wLi = log_wL_arr[il]
        
        k_ = (vii - v_min) / dv
        k0 = int(k_)
        k1 = k0 + 1
        av = k_ - k0

        l_ = (log_wGi - log_wG_min) / dxG
        l0 = int(l_)
        l1 = l0 + 1
        awG = l_ - l0

        m_ = (log_wLi - log_wL_min) / dxL
        m0 = int(m_)
        m1 = m0 + 1
        awL = m_ - m0

        if k0 >= 0: #TODO: why is this here again? A:...
            
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

    #There are still some non-zero elements in the circular buffer, add them too.
    for j in range(1, spread_size):
        kj = k + j
        kj_c = kj & circ_mask

        #Iterate over the wG x wL grid; if value in the circular
        #buffer is non-zero add it to the block memory.
        #TO-DO: this should be compiled into a memcpy.
        S_klm[kj] = S_circ[kj_c]
        S_circ[kj_c] = 0.0
        
        #for l in range(NwG):
        #    for m in range(NwL):
        #        val_klm = S_circ[kj_c, l, m]
        #        if val_klm != 0.0:
        #            
        #            S_klm[kj, l, m] = val_klm 
        #            S_circ[kj_c, l, m] = 0.0











        
        
        
        
        
        
        
        