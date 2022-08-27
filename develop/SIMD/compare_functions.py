import sys
sys.path.append('../../demo')
import numpy as np
import matplotlib.pyplot as plt
from HITEMP_spectra import init_database, calc_stick_spectrum
from discrete_integral_transform_simd import (
    calc_matrix_py1,
    calc_matrix_py2,
    calc_matrix_cy1,
    calc_matrix_cpp1,
    calc_matrix_simd1,
    apply_transform_py1,
    synthesize_spectrum,
    )

HITEMP_path = "C:/HITEMP/"
init_database([HITEMP_path + "02_2000-2125_HITEMP2010.par",
               HITEMP_path + "02_2125-2250_HITEMP2010.par",
               HITEMP_path + "02_2250-2500_HITEMP2010.par"])

v_min = 2000.0 #cm-1
v_max = 2400.0 #cm-1
dv =     0.002 #cm-1
v_arr = np.arange(v_min,v_max,dv, dtype=np.float32) #cm-1

T = 1500.0 #K
p =    0.1 #bar

v0i,log_wGi,log_wLi,S0i = calc_stick_spectrum(p,T)
print('{:.2f}M lines loaded...'.format(len(v0i)*1e-6))
print('              Matrix (ms):  Transform (ms):')

print('C++ add_at:       ', end='')
I1_arr, S_klm, tl = synthesize_spectrum(v_arr, v0i, log_wGi, log_wLi, S0i,
                        f_calc_matrix=calc_matrix_cpp1,
                        f_apply_transform=apply_transform_py1)
print('{:10.0f}\t{:10.0f}'.format((tl[1] - tl[0])*1e3, (tl[2] - tl[1])*1e3))

print('SIMD calc_matrix: ', end='')
I2_arr, S_klm, tl = synthesize_spectrum(v_arr, v0i, log_wGi, log_wLi, S0i,
                        f_calc_matrix=calc_matrix_simd1,
                        f_apply_transform=apply_transform_py1)
print('{:10.0f}\t{:10.0f}'.format((tl[1] - tl[0])*1e3, (tl[2] - tl[1])*1e3))


for i in range(5):
    print('Baseline:      ',end='')
    I_arr, S_klm, tl = synthesize_spectrum(v_arr, v0i, log_wGi, log_wLi, S0i,
                            f_calc_matrix=calc_matrix_py1,
                            f_apply_transform=apply_transform_py1)
    print('{:10.0f}\t{:10.0f}'.format(
        (tl[1] - tl[0])*1e3, (tl[2] - tl[1])*1e3))
print('')

for i in range(5):
    print('Cython add_at: ', end='')
    I_arr, S_klm, tl = synthesize_spectrum(v_arr, v0i, log_wGi, log_wLi, S0i,
                            f_calc_matrix=calc_matrix_cy1,
                            f_apply_transform=apply_transform_py1)
    print('{:10.0f}\t{:10.0f}'.format(
        (tl[1] - tl[0])*1e3, (tl[2] - tl[1])*1e3))
print('')

for i in range(5):
    print('SIMD calc_matrix: ', end='')
    I_arr, S_klm, tl = synthesize_spectrum(v_arr, v0i, log_wGi, log_wLi, S0i,
                            f_calc_matrix=calc_matrix_simd1,
                            f_apply_transform=apply_transform_py1)
    print('{:10.0f}\t{:10.0f}'.format(
        (tl[1] - tl[0])*1e3, (tl[2] - tl[1])*1e3))
print('')


plt.axhline(0,c='k')
plt.plot(v_arr, I1_arr, 'k', lw=3)
plt.plot(v_arr, I2_arr,  'r', lw=1)
##plt.plot(v_arr, I1_arr - I2_arr)
plt.xlim(v_max, v_min)
plt.show()
