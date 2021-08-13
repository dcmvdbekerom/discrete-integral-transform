import numpy as np
import matplotlib.pyplot as plt
from ditlog import synthesize_spectrum as spectrum

gLv  = lambda v,v0,w: 2/(np.pi*w) * 1 / (1 + 4*((v-v0)/w)**2)

N = 20000
v0 = 2100.0
w = 1.0

dxv = 1e-5
dv = v0*dxv

v_lin = v0 + np.arange(-N//2,N//2)*dv
v_log = v0*np.exp(np.arange(-N//2,N//2)*dxv)

I_lin = gLv(v_lin,v0,w)
I_log = gLv(v_log,v0,w)

I_ditlin = spectrum(v_lin,
                    np.array([v0]),
                    np.array([np.log(w)]),
                    np.array([1.0]),
                    dxv,
                    folding_thresh = 1e-6
                    )

I_ditlog = spectrum(v_log,
                    np.array([v0]),
                    np.array([np.log(w)]),
                    np.array([1.0]),
                    dxv,
                    folding_thresh = 1e-6
                    )

##plt.plot(I_ditlin - I_lin, label = 'ditlin-lin')
plt.plot(I_ditlog, label = 'ditlog')
plt.plot(I_log, '--', label = 'log')


plt.legend()
##plt.yscale('log')
plt.show()
