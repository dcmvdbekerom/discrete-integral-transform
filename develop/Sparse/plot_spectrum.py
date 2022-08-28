
from discrete_integral_transform import synthesize_spectrum
import sys
sys.path.append('../../demo/')
from HITEMP_spectra import init_database, calc_stick_spectrum
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
dv =     0.002 #cm-1
v_arr = np.arange(v_min,v_max,dv) #cm-1

dxG = 0.1
dxL = 0.1

T = 2000.0 #K
p =    0.1 #bar

v0i,log_wGi,log_wLi,S0i = calc_stick_spectrum(p,T)
wGi = np.exp(log_wGi)
wLi = np.exp(log_wLi)
print('{:.2f}M lines '.format(len(v0i)*1e-6))

I_arr, S_klm = synthesize_spectrum(v_arr,v0i,log_wGi,log_wLi,S0i,dxG=dxG,dxL=dxL)

##sparsity = np.log(np.sum(S_klm!=0.0,0)/S_klm.shape[0])

sparsity = np.sum(S_klm,0)/(np.sum(S_klm**2,0)**0.5)


plt.imshow(sparsity, clim=(0,1))
plt.show()


##plt.plot(v_arr,I_arr,'-',lw=0.5)
##plt.xlim(v_arr[-1],v_arr[0])
##plt.axhline(0,c='k')
##plt.xlabel('$\\nu$ (cm$^{-1}$)')
##plt.xlim(2231.01,2228.99)
##plt.ylim(-0.02,0.3)
##plt.show()
