import sys
sys.path.append('../../demo')
import numpy as np
import matplotlib.pyplot as plt
from HITEMP_spectra import init_database, calc_stick_spectrum
from functools import partial
import pyfftw
from discrete_integral_transform_simd import (
    calc_matrix_py1,
    calc_matrix_py2,
    calc_matrix_cy0_222,
    calc_matrix_cy0_333,
    calc_matrix_cy1,
    #calc_matrix_cy2,
    calc_matrix_cpp1,
    calc_matrix_cpp2,
    calc_matrix_simd1,
    apply_transform_py1,
    apply_transform_py2,
    synthesize_spectrum,
    plan_FFTW,
    )

HITEMP_path = "C:/HITEMP/"
init_database([HITEMP_path + "02_2000-2125_HITEMP2010.par",
               HITEMP_path + "02_2125-2250_HITEMP2010.par",
               HITEMP_path + "02_2250-2500_HITEMP2010.par"])

v_min = 2001.0 #cm-1
v_max = 2400.0 #cm-1
dv =     0.002 #cm-1
v_arr = np.arange(v_min,v_max,dv, dtype=np.float32) #cm-1


T = 1500.0 #K
p =    0.1 #bar


##v0i, dai = np.load('CO2_hitemp.npy')[:2]
##_,log_wGi,log_wLi,S0i = calc_stick_spectrum(p,T)
v0i,log_wGi,log_wLi,S0i = calc_stick_spectrum(p,T)

##da_min = np.min(dai)
##da_max = np.max(dai)
##v_diff_min, v_diff_max = p * da_min, p * da_max
##k_diff_min, k_diff_max = int(np.floor(v_diff_min / dv)), int(np.ceil(v_diff_max / dv))
##spread_size = k_diff_max - k_diff_min + 1 + 1 # +1 to accomodate k0 + 1
##extra_params = [dai, p, k_diff_min, spread_size]



print('{:.2f}M lines loaded...'.format(len(v0i)*1e-6))

##with open('wisdom.txt','r') as f:
##    wisdom = pickle.load(f)
dxG = 0.2
dxL = 0.2

##synthesize_spectrum(v_arr, v0i, log_wGi, log_wLi, S0i, dxG=dxG, dxL=dxL, extra_params=extra_params, plan_only=True)

for i in range(5):
    print('Baseline:  ',end='')
    I1_arr, S_klm, tl = synthesize_spectrum(v_arr, v0i, log_wGi, log_wLi, S0i, dxG=0.1, dxL=0.1,
                                            f_calc_matrix=calc_matrix_cy0_222,
                                            f_apply_transform=apply_transform_py1,
                                            #extra_params=extra_params,
                                            )
    print('{:4.0f}, {:4.0f}'.format(
        (tl[1] - tl[0])*1e3, (tl[2] - tl[1])*1e3))
print('')

for i in range(5):
    print('Cython:    ', end='')
    I2_arr, S_klm, tl = synthesize_spectrum(v_arr, v0i, log_wGi, log_wLi, S0i, dxG=dxG, dxL=dxL,
                                            f_calc_matrix=calc_matrix_cy0_333,
                                            f_apply_transform=apply_transform_py1,
                                            #extra_params=extra_params,
                                            )
    
    print('{:4.0f}, {:4.0f} -- '.format(
        (tl[1] - tl[0])*1e3, (tl[2] - tl[1])*1e3))
print('')

##for i in range(5):
##    print('C++:    ', end='')
##    I2_arr, S_klm, tl = synthesize_spectrum(v_arr, v0i, log_wGi, log_wLi, S0i,
##                            f_calc_matrix=calc_matrix_cpp2,
##                            f_apply_transform=apply_transform_py1)
##    print('{:10.0f}\t{:10.0f}'.format(
##        (tl[1] - tl[0])*1e3, (tl[2] - tl[1])*1e3))
##print('')

fig,ax = plt.subplots(2, sharex=True)
ax[0].axhline(0,c='k')
ax[0].plot(v_arr, I1_arr, 'k', lw=3, label = 'Baseline')
ax[0].plot(v_arr, I2_arr,  'r', lw=1, label = 'Cython')
ax[0].set_xlim(v_max, v_min)
ax[0].legend(loc=1)

ax[1].plot(v_arr, I1_arr - I2_arr)
plt.show()
