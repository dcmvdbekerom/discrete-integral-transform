import sys
sys.path.append('../../demo')
import numpy as np
import matplotlib.pyplot as plt
from HITEMP_spectra import init_database, calc_stick_spectrum
from discrete_integral_transform_simple import (
    calc_matrix_py1,
    apply_transform_py1,
    synthesize_spectrum,
    )

HITEMP_path = "C:/HITEMP/"
init_database([HITEMP_path + "02_2000-2125_HITEMP2010.par",
               HITEMP_path + "02_2125-2250_HITEMP2010.par",
               HITEMP_path + "02_2250-2500_HITEMP2010.par"])

v_min = 2200.0 #cm-1
v_max = 2400.0 #cm-1
dv =     0.002 #cm-1
v_arr = np.arange(v_min,v_max,dv) #cm-1

T = 1500.0 #K
p =    0.1 #bar

v0i,log_wGi,log_wLi,S0i = calc_stick_spectrum(p,T)
print('{:.2f}M lines loaded...'.format(len(v0i)*1e-6))

I_arr, S_klm = synthesize_spectrum(v_arr,v0i,log_wGi,log_wLi,S0i)

##plt.axhline(0,c='k')
##plt.plot(v_arr, I_arr, lw=0.5)
##plt.xlim(v_max, v_min)
##plt.show()
