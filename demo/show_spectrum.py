from HITEMP_spectra import init_database, calc_stick_spectrum
from discrete_integral_transform import synthesize_spectrum
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
v = np.arange(v_min,v_max,dv) #cm-1

dxG = 0.2
dxL = 0.2

T = 1000.0 #K
p =    0.1 #bar




v0i,log_wGi,log_wLi,S0i = calc_stick_spectrum(p,T)
print('{:.2f}M lines '.format(len(v0i)*1e-6))
I0,S_klm,J = synthesize_spectrum(v,v0i,log_wGi,log_wLi,S0i,dxG=dxG,dxL=dxL,optimized=False)

Iexp = I0 + np.random.normal(0,0.05,I0.size)
I1,S_klm,J = synthesize_spectrum(v,v0i,log_wGi,log_wLi,S0i,Iexp=Iexp,dxG=dxG,dxL=dxL,optimized=False)


plt.plot(v,Iexp,'.')
plt.plot(v,I0,'-')

plt.xlim(v_max,v_min)
plt.grid(True,alpha=0.2)
plt.axhline(0,c='k',lw=1)
plt.show()
