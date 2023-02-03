import sys
sys.path.append('../../demo')
import numpy as np
import matplotlib.pyplot as plt
from HITEMP_spectra import load_HITRAN, init_database, calc_stick_spectrum
from discrete_integral_transform_sparse import init_w_axis, calc_matrix
from functools import partial
from scipy.signal._signaltools import _calc_oa_lens
import pyfftw

HITEMP_path = "C:/HITEMP/"
init_database([HITEMP_path + "02_2000-2125_HITEMP2010.par",
               HITEMP_path + "02_2125-2250_HITEMP2010.par",
               HITEMP_path + "02_2250-2500_HITEMP2010.par"])

v_min = 2000.0 #cm-1
v_max = 2400.0 #cm-1
dv =     0.002 #cm-1
v_arr = np.arange(v_min,v_max,dv, dtype=np.float32) #cm-1

T = 1500.0 #K
p =    4.0 #bar

v0_arr, da_arr = np.load('CO2_hitemp.npy')[:2]
_,log_wG_arr,log_wL_arr,S0_arr = calc_stick_spectrum(p,T)
print('{:.2f}M lines loaded...'.format(len(v0_arr)*1e-6))

Nlen = 5000

S0_arr = S0_arr[:Nlen]
v0_arr = v0_arr[:Nlen]
da_arr = da_arr[:Nlen]
log_wG_arr = log_wG_arr[:Nlen]
log_wL_arr = log_wL_arr[:Nlen]

da_min = np.min(da_arr)
da_max = np.max(da_arr)
vi_arr = v0_arr + p*da_arr

dxG = 0.1
dxL = 0.4

log_wG = init_w_axis(dxG,log_wG_arr)
log_wL = init_w_axis(dxL,log_wL_arr)


v_min = v_arr[0]
log_wG_min = log_wG[0]
log_wL_min = log_wL[0]

v_diff_min, v_diff_max = p * da_min, p * da_max
k_diff_min, k_diff_max = int(np.floor(v_diff_min / dv)), int(np.ceil(v_diff_max / dv))


##def calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i):
##    
##    S_klm = np.zeros((2 * v.size, log_wG.size, log_wL.size))
##    
##    ki0, ki1, avi = get_indices2(v0i, v, 0.002)          #Eqs 3.4 & 3.6
##    li0, li1, aGi = get_indices2(log_wGi, log_wG, 0.1) #Eqs 3.7 & 3.10
##    mi0, mi1, aLi = get_indices2(log_wLi, log_wL, 0.4) #Eqs 3.7 & 3.10
##    
##    np.add.at(S_klm, (ki0, li0, mi0), S0i * (1-avi) * (1-aGi) * (1-aLi))
##    np.add.at(S_klm, (ki0, li0, mi1), S0i * (1-avi) * (1-aGi) *    aLi )
##    np.add.at(S_klm, (ki0, li1, mi0), S0i * (1-avi) *    aGi  * (1-aLi))
##    np.add.at(S_klm, (ki0, li1, mi1), S0i * (1-avi) *    aGi  *    aLi )
##    np.add.at(S_klm, (ki1, li0, mi0), S0i *    avi  * (1-aGi) * (1-aLi))
##    np.add.at(S_klm, (ki1, li0, mi1), S0i *    avi  * (1-aGi) *    aLi )
##    np.add.at(S_klm, (ki1, li1, mi0), S0i *    avi  *    aGi  * (1-aLi))
##    np.add.at(S_klm, (ki1, li1, mi1), S0i *    avi  *    aGi  *    aLi )
##    
##    return S_klm


def calc_matrix_ref(v_arr, log_wG, log_wL, v0_arr, log_wG_arr, log_wL_arr, S0_arr):
    S_klm = np.zeros((2*v_arr.size, log_wG.size, log_wL.size))

    for il in range(v0_arr.size):

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

        if k0 >= 0:

            S_klm[k0, l0, m0] += (1-av) * (1-awG) * (1-awL) * S0i
            S_klm[k0, l0, m1] += (1-av) * (1-awG) *    awL  * S0i
            S_klm[k0, l1, m0] += (1-av) *    awG  * (1-awL) * S0i
            S_klm[k0, l1, m1] += (1-av) *    awG  *    awL  * S0i
            S_klm[k1, l0, m0] +=    av  * (1-awG) * (1-awL) * S0i
            S_klm[k1, l0, m1] +=    av  * (1-awG) *    awL  * S0i
            S_klm[k1, l1, m0] +=    av  *    awG  * (1-awL) * S0i
            S_klm[k1, l1, m1] +=    av  *    awG  *    awL  * S0i

    return S_klm



def calc_matrix_circ(v_arr, log_wG, log_wL, v0_arr, log_wG_arr, log_wL_arr, S0_arr):

    Nv = v_arr.size
    NwG = log_wG.size
    NwL = log_wL.size

    spread_size = k_diff_max - k_diff_min + 1 + 1 # +1 to accomodate k0 + 1
    circ_buffer_size = int(2**np.ceil(np.log2(spread_size)))
    circ_mask = circ_buffer_size - 1 #if spread len is 0b01000, circ_mask is 0b00111

    S_circ = np.zeros((circ_buffer_size, NwG, NwL))
    S_klm = np.zeros((2*Nv, NwG, NwL))

    iv_old = 0
    for il in range(len(v0_arr)):

        v0i = v0_arr[il]
        iv = int((v0_arr[il] - v_min) / dv) 
        if iv != iv_old:
            
            k = iv_old + k_diff_min
            iv_step = iv - iv_old

            #The index kj is the index of the v-grid.
            #Add the circular buffer to the block as much as needed
            #to catch up with the change in iv.
            for j in range(np.min((iv_step, spread_size))):
                kj = k + j
                kj_c = kj & circ_mask

                #Iterate over the wG x wL grid; if value in the circular
                #buffer is non-zero add it to the block memory.
                #TO-DO: this should be compiled into a memcpy.
                for l in range(NwG):
                    for m in range(NwL):
                        val_klm = S_circ[kj_c, l, m]
                        if val_klm != 0.0:
                            
                            S_klm[kj, l, m] = val_klm 
                            S_circ[kj_c, l, m] = 0.0

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
        for l in range(NwG):
            for m in range(NwL):
                val_klm = S_circ[kj_c, l, m]
                if val_klm != 0.0:
                    
                    S_klm[kj, l, m] = val_klm 
                    S_circ[kj_c, l, m] = 0.0

    return S_klm


S_klm_ref  = calc_matrix_ref (v_arr, log_wG, log_wL, v0_arr, log_wG_arr, log_wL_arr, S0_arr)
S_klm_circ = calc_matrix_circ(v_arr, log_wG, log_wL, v0_arr, log_wG_arr, log_wL_arr, S0_arr)

S_klm_ref0  = np.sum(np.sum(S_klm_ref,  axis=2), axis=1)
S_klm_circ0 = np.sum(np.sum(S_klm_circ, axis=2), axis=1)

plt.plot(v_arr, S_klm_ref [:v_arr.size,0,0])
plt.plot(v_arr, S_klm_circ[:v_arr.size,0,0],'k--')
plt.xlim(v0_arr[0],v0_arr[-1])

##plt.plot(S_klm[:,0,0]-S_klm_ref[:,0,0],'o-')


plt.show()

