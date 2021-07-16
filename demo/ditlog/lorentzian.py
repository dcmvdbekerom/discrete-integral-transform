import numpy as np
import matplotlib.pyplot as plt
from ditlog import synthesize_spectrum as spectrum

gLv  = lambda v,v0,w: 2/(np.pi*w) * 1 / (1 + 4*((v-v0)/w)**2)

N = 200
v0 = 2100.0
w = 1.0

dxv = 1e-3
dv = v0*dxv

v_lin = v0 + np.arange(-N//2,N//2)*dv
v_log = v0*np.exp(np.arange(-N//2,N//2)*dxv)

plt.plot(v_lin, gLv(v_lin,v0,w),'-', label = 'linear grid')

for zp in [2,4,8,16]:
    I_ditlog = spectrum(v_lin,
                        np.array([v0]),
                        np.array([np.log(w)]),
                        np.array([1.0]),
                        dxv,
                        zero_pad = zp,
                        folding_thresh = 1e-6
                        )

    plt.plot(v_lin, I_ditlog, '--', label='ditlog ({:d}x)'.format(zp))

plt.legend()
plt.yscale('log')
plt.show()
