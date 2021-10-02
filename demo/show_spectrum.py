from HITEMP_spectra import init_database, calc_stick_spectrum
from discrete_integral_transform_simple import synthesize_spectrum
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
v_arr = np.arange(v_min,v_max,dv) #cm-1

dxG = 0.1
dxL = 0.1

T = 1000.0 #K
p =    0.1 #bar

v0i,log_wGi,log_wLi,S0i = calc_stick_spectrum(p,T)
print('{:.2f}M lines '.format(len(v0i)*1e-6))

I_arr, S_kl = synthesize_spectrum(v_arr,v0i,log_wGi,log_wLi,S0i,dxG=dxG,dxL=dxL)
plt.plot(v_arr,I_arr,'-')
plt.xlim(v_arr[-1],v_arr[0])
plt.show()
