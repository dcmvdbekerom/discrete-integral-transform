import HITEMP_spectra
import fss_py
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

HITEMP_spectra.init_database([])
W = fss_py.W
v_min = 2200.0 #cm-1
v_max = 2400.0 #cm-1
dv =     0.001 #cm-1
v = np.arange(v_min,v_max,dv) #cm-1

T = 1000.0 #K
p =    0.1 #bar

v0i,log_wGi,log_wLi,S0i = HITEMP_spectra.calc_stick_spectrum(p,T)

I0,S_klm = fss_py.spectrum(v,
                           0.02,
                           0.02,
                           v0i,
                           log_wGi,
                           log_wLi,
                           S0i,
                           awL_kind = 'linear',
                           awG_kind = 'linear',)

np.save('I0.npy',I0)
np.save('S_klm.npy',S_klm)

I0 = np.load('I0.npy')
S_klm = np.load('S_klm.npy')



I1,S_klm = fss_py.spectrum(v,
                           0.1,
                           0.1,
                           v0i,
                           log_wGi,
                           log_wLi,
                           S0i,
                           awG_kind = 'linear',
                           awL_kind = 'linear',)

I2,S_klm = fss_py.spectrum(v,
                           0.1,
                           0.1,
                           v0i,
                           log_wGi,
                           log_wLi,
                           S0i,
                           awG_kind = 'opti',
                           awL_kind = 'opti',)


print(S_klm.shape)


plt.xlim(v_max,v_min)
plt.grid(True,alpha=0.2)
plt.axhline(0,c='k',lw=1)

plt.plot(v,np.abs(I1-I0),lw=2,label='linear')
plt.plot(v,np.abs(I2-I0),lw=2,label='optimized')

print('linear:',np.sum((I1-I0)**2)**0.5)
print('optimized:  ',np.sum((I2-I0)**2)**0.5)

plt.legend()
plt.show()
