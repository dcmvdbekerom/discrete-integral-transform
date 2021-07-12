from HITEMP_spectra import init_database, calc_stick_spectrum
from discrete_integral_transform import synthesize_spectrum
from discrete_integral_transform_log import synthesize_spectrum as synthesize_spectrum_log
from scipy.interpolate import CubicSpline
##from dit_log import synthesize_spectrum as synthesize_spectrum_log
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import sys

## Download database files from https://hitran.org/hitemp/

HITEMP_path = "C:/HITEMP/"
init_database([HITEMP_path + "02_2000-2125_HITEMP2010.par",
               HITEMP_path + "02_2125-2250_HITEMP2010.par",
               HITEMP_path + "02_2250-2500_HITEMP2010.par"])

v_min = 2200.0 #cm-1
v_max = 2400.0 #cm-1
dv =     0.002 #cm-1
v_lin = np.arange(v_min,v_max,dv) #cm-1

dxG = 0.01
dxL = 0.01

Nv = 100000
dxv = 1e-6

T = 1000.0 #K
p =    0.1 #bar

log_v = np.log(v_min) + np.arange(Nv) * dxv
v_log = np.exp(log_v)

#calculate line params
v0i,log_wGi,log_wLi,S0i = calc_stick_spectrum(p,T)
log_v0i = np.log(v0i)
print('{:.2f}M lines '.format(len(v0i)*1e-6))

# linear v-grid:
I0_lin, S_klm_lin = synthesize_spectrum(v_lin,v0i,log_wGi,log_wLi,S0i,dxG=dxG,dxL=dxL,optimized=False)

# log v-grid:
I0_log, S_klm_lin = synthesize_spectrum_log(v_log,v0i,log_wGi,log_wLi,S0i,dxG=dxG,dxL=dxL)

fig,ax = plt.subplots(2,1,sharex=True)

ax[0].plot(v_lin,I0_lin,'.', label='linear')
ax[0].plot(v_log,I0_log,'.',label='log')
##ax[0].set_xlim(2305,2303)
ax[0].legend()

cs = CubicSpline(v_lin,I0_lin)
idx = v_log < v_max
ax[1].plot(v_log[idx], I0_log[idx] - cs(v_log[idx]),label='log - linear (intp.)')
plt.show()
