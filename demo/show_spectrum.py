from HITEMP_spectra import init_database, calc_stick_spectrum
from discrete_integral_transform import synthesize_spectrum
from discrete_integral_transform_log import synthesize_spectrum as synthesize_spectrum_log
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


## Download database files from https://hitran.org/hitemp/

HITEMP_path = "C:/HITEMP/"
init_database([HITEMP_path + "02_2000-2125_HITEMP2010.par",
               HITEMP_path + "02_2125-2250_HITEMP2010.par",
               HITEMP_path + "02_2250-2500_HITEMP2010.par"])

v_min = 2200.0 #cm-1
v_max = 2400.0 #cm-1
dv =     0.0002 #cm-1
v_lin = np.arange(v_min,v_max,dv) #cm-1

dxG = 0.2
dxL = 0.2

Nv = 1000000
dxv = 1e-7

T = 1000.0 #K
p =    0.1 #bar

#calculate line params
v0i,log_wGi,log_wLi,S0i = calc_stick_spectrum(p,T)
log_v0i = np.log(v0i)
print('{:.2f}M lines '.format(len(v0i)*1e-6))

# linear v-grid:
I0_lin, S_klm_lin = synthesize_spectrum(v_lin,v0i,log_wGi,log_wLi,S0i,dxG=dxG,dxL=dxL,optimized=False)

# log v-grid:
log_v, I0_log, S_klm_log = synthesize_spectrum_log(log_v0i,log_wGi,log_wLi,S0i,
                                        np.log(v_min), Nv, dxv,
                                        dxG=dxG,dxL=dxL)
v_log = np.exp(log_v)


fig,ax = plt.subplots(2,1,sharex=True)

ax[0].plot(v_lin,I0_lin,'-', label='linear')
ax[0].plot(v_log,I0_log,'--',label='log')
ax[0].set_xlim(v_max,v_min)
ax[0].legend()

ax[1].plot(v_log, I0_log - np.interp(v_log,v_lin,I0_lin),label='log - linear (intp.)')
plt.show()
