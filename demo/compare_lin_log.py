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

T = 1000.0 #K
p =    0.1 #bar

dxG = 0.1
dxL = 0.1

#calculate line params
v0i,log_wGi,log_wLi,S0i = calc_stick_spectrum(p,T)
print('{:.2f}M lines '.format(len(v0i)*1e-6))


#init v-grids:

v_min = 2200.0 #cm-1
v_max = 2400.0 #cm-1
dv = 0.001 #cm-1
v_lin = np.arange(v_min,v_max,dv)

R = v_max/dv
Nv_log = int(np.log(v_max/v_min)*R)
v_log = v_min * np.exp(np.arange(Nv_log) / R)

# The log-version should also be compatible with linear inputs.
# Let's double check whether that's the case:

I_lin, S_klm = synthesize_spectrum(v_lin,
                                   v0i, log_wGi, log_wLi, S0i,
                                   dxG=dxG, dxL=dxL,
                                   optimized=False)

##I_lin2, S_klm = synthesize_spectrum_log(v_lin,
##                                        v0i, log_wGi, log_wLi, S0i,
##                                        dxG=dxG, dxL=dxL,
##                                        log_correction=False)
##
##plt.plot(v_lin,I_lin,label='I_lin')
##plt.plot(v_lin,np.abs(I_lin-I_lin2), label='I_lin-I_lin2')
##plt.xlim(v_max,v_min)
##plt.yscale('log')
##plt.legend()
##plt.show()

# Now we compare it to the log-grid:
I_log, S_klm = synthesize_spectrum_log(v_log,
                                       v0i, log_wGi, log_wLi, S0i,
                                       dxG=dxG, dxL=dxL,
                                       log_correction=False)

##plt.plot(v_lin,I_lin, label='I_lin')
##plt.plot(v_log,I_log, label='I_log')
##plt.xlim(v_max,v_min)
##plt.yscale('log')
##plt.legend()
##plt.show()

# Finally, let's see if the asymmetry correction does anything:

I_log2, S_klm = synthesize_spectrum_log(v_log,
                                        v0i, log_wGi, log_wLi, S0i,
                                        dxG=dxG, dxL=dxL,
                                        log_correction=True)

I_loglin = CubicSpline(v_log, I_log)(v_lin)
##plt.plot(v_lin,I_lin, '-', label='I_lin')
##plt.plot(v_log,I_log, '-', label='I_log')
##plt.plot(v_lin,I_loglin, '-', label='I_loglin')
##plt.plot(v_log,I_log2, '-',label='I_log2')
plt.plot(v_lin,I_loglin - I_lin)
plt.plot(v_log, I_log2 - I_log)



plt.xlim(v_max,v_min)
##plt.yscale('log')
plt.legend()
plt.show()




