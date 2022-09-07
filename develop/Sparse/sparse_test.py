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

Nlen = -1#5000

S0_arr = S0_arr[:Nlen]
v0_arr = v0_arr[:Nlen]
da_arr = da_arr[:Nlen]
log_wG_arr = log_wG_arr[:Nlen]
log_wL_arr = log_wL_arr[:Nlen]

da_min = np.min(da_arr)
da_max = np.max(da_arr)
vi_arr = v0_arr + p*da_arr


dxG = 0.05
dxL = 0.1

log_wG = init_w_axis(dxG,log_wG_arr)
log_wL = init_w_axis(dxL,log_wL_arr)


##S_klm_ref = calc_matrix(v_arr, log_wG, log_wL,
##                        vi_arr[valid],
##                        log_wG_arr[valid],
##                        log_wL_arr[valid],
##                        S0_arr[valid])
##
##S_klm_ref0 = np.sum(np.sum(S_klm_ref,axis=2),axis=1)
##
##
##
##
####plt.plot(v_arr,S_klm_ref[:v_arr.size,0,0])
##plt.plot(v_arr,S_klm_ref0[:v_arr.size])
####plt.vlines(vi_arr,0,S0_arr)
##plt.xlim(v_arr[0],v_arr[0]+0.1)
####plt.show()
####sys.exit()

##########################################

wstep = dv


Nv = v_arr.size
NwG = log_wG.size
NwL = log_wL.size

v_min = v_arr[0]
log_wG_min = log_wG[0]
log_wL_min = log_wL[0]

truncation = 0.5 #cm-1

lineshape_size = 2*int(truncation/wstep) #left-to-right

(block_size, 
overlap_size, 
in1_step, 
in2_step) = _calc_oa_lens(Nv, lineshape_size)

v_diff_min, v_diff_max = p * da_min, p * da_max
k_diff_min, k_diff_max = int(np.floor(v_diff_min / dv)), int(np.ceil(v_diff_max / dv))

spread_size = k_diff_max - k_diff_min + 1 + 1 # +1 to accomodate k0 + 1
circ_buffer_size = int(2**np.ceil(np.log2(spread_size)))
circ_mask = circ_buffer_size - 1 #if spread len is 0b01000, circ_mask is 0b00111

S_circ = np.zeros((circ_buffer_size,NwG,NwL))
S_klm = np.zeros((block_size,NwG,NwL))
S_klm_ref = np.zeros((2*Nv,NwG,NwL))

latest_line = np.zeros((NwG,NwL), dtype=int)
cum_skip = np.zeros((NwG,NwL), dtype=int)

print(vi_arr[0], vi_arr[-1])
print(spread_size, lineshape_size, block_size)

indices = []
iv_old = 0
for il in range(len(v0_arr)):

    v0i = v0_arr[il]
    iv = int((v0_arr[il] - v_min) / dv) 
    if iv != iv_old:
        
        k = iv_old + k_diff_min
        iv_step = iv - iv_old

        for j in range(np.min((iv_step, spread_size))):
            kj = k + j
            kj_c = kj & circ_mask
            
            for l in range(NwG):
                for m in range(NwL):
                    
                    val_klm = S_circ[kj_c, l, m]
                    if val_klm != 0.0:

                        if kj - latest_line[l,m] > lineshape_size:
                            cum_skip[l,m] += kj - latest_line[l,m] - lineshape_size
                            print(l,m,'moved a line!')
                            # TO-DO: update skip_index_arr
                            
                        kj_adj = kj - cum_skip[l,m]
                        S_klm[kj_adj, l, m] = val_klm 
                        S_circ[kj_c, l, m] = 0.0
                        latest_line[l, m] = kj

        iv_old = iv
        indices = []
        
    #continue adding lines to circular buffer
    S0i = S0_arr[il]
    vii = v0_arr[il] + p * da_arr[il]
    log_wGi = log_wG_arr[il]
    log_wLi = log_wL_arr[il]
    
    k = (vii - v_min) / dv
    k0 = int(k)
    k1 = k0 + 1
    av = k - k0

    l = (log_wGi - log_wG_min) / dxG
    l0 = int(l)
    l1 = l0 + 1
    awG = l - l0

    m = (log_wLi - log_wL_min) / dxL
    m0 = int(m)
    m1 = m0 + 1
    awL = m - m0

    if k0 >= 0:

        indices.append(k0)
        indices.append(k1)

        S_klm_ref[k0, l0, m0] += (1-av) * (1-awG) * (1-awL) * S0i
        S_klm_ref[k0, l0, m1] += (1-av) * (1-awG) *    awL  * S0i
        S_klm_ref[k0, l1, m0] += (1-av) *    awG  * (1-awL) * S0i
        S_klm_ref[k0, l1, m1] += (1-av) *    awG  *    awL  * S0i
        S_klm_ref[k1, l0, m0] +=    av  * (1-awG) * (1-awL) * S0i
        S_klm_ref[k1, l0, m1] +=    av  * (1-awG) *    awL  * S0i
        S_klm_ref[k1, l1, m0] +=    av  *    awG  * (1-awL) * S0i
        S_klm_ref[k1, l1, m1] +=    av  *    awG  *    awL  * S0i
        
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

S_klm0 = np.sum(np.sum(S_klm,axis=2),axis=1)
S_klm_ref0 = np.sum(np.sum(S_klm_ref,axis=2),axis=1)

##valid = vi_arr >= v_arr[0]
##S_klm_ref2 = calc_matrix(v_arr, log_wG, log_wL,
##                        vi_arr[valid],
##                        log_wG_arr[valid],
##                        log_wL_arr[valid],
##                        S0_arr[valid])


plt.plot(v_arr[:S_klm.shape[0]],S_klm_ref[:S_klm.shape[0],0,0])
##plt.plot(v_arr,S_klm_ref2[:v_arr.size,0,0],'k--')
plt.plot(v_arr[:S_klm.shape[0]], S_klm[:,0,0],'k--')


##plt.plot(S_klm[:,0,0]-S_klm_ref[:S_klm.shape[0],0,0],'o-')

plt.xlim(v0_arr[0],v0_arr[-1])
plt.show()

