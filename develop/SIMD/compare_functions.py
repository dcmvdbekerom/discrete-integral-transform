import sys
sys.path.append('../../demo/')
from HITEMP_spectra import init_database, calc_stick_spectrum
from discrete_integral_transform_simple import (init_w_axis,
                                                calc_matrix as py_calc_matrix,
                                                apply_transform as py_apply_transform)
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def synthesize_spectrum(v, v0i, log_wGi, log_wLi, S0i,
                        dxG = 0.1, dxL = 0.1,
                        f_calc_matrix=py_calc_matrix,
                        f_apply_transform=py_apply_transform):
    
    idx = (v0i >= np.min(v)) & (v0i < np.max(v))
    v0i, log_wGi, log_wLi, S0i = v0i[idx], log_wGi[idx], log_wLi[idx], S0i[idx]
    log_wG = init_w_axis(dxG,log_wGi) #Eq 3.8
    log_wL = init_w_axis(dxL,log_wLi) #Eq 3.9

    t0 = perf_counter()
    S_klm = f_calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i)
    t1 = perf_counter()
    print('Calc matrix:     {:10.3f} ms'.format((t1 - t0)*1e3))

    t1 = perf_counter()
    I = f_apply_transform(v, log_wG, log_wL, S_klm)
    t2 = perf_counter()
    print('Apply transform: {:10.3f} ms'.format((t2 - t1)*1e3))
    return I, S_klm


## Download database files from https://hitran.org/hitemp/
HITEMP_path = "C:/HITEMP/"
init_database([HITEMP_path + "02_2000-2125_HITEMP2010.par",
               HITEMP_path + "02_2125-2250_HITEMP2010.par",
               HITEMP_path + "02_2250-2500_HITEMP2010.par"])

v_min = 2000.0 #cm-1
v_max = 2400.0 #cm-1
dv =     0.002 #cm-1
v_arr = np.arange(v_min,v_max,dv) #cm-1

T = 1500.0 #K
p =    0.1 #bar

v0i,log_wGi,log_wLi,S0i = calc_stick_spectrum(p,T)
print('{:.2f}M lines loaded...'.format(len(v0i)*1e-6))

I_arr, S_klm = synthesize_spectrum(v_arr,v0i,log_wGi,log_wLi,S0i)
