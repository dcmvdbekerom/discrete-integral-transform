import numpy as np
import matplotlib.pyplot as plt
from ditlog import synthesize_spectrum as spectrum

gLv  = lambda v,v0,w: 2/(np.pi*w) * 1 / (1 + 4*((v-v0)/w)**2)


vmin = 2100.0
vmax = 2300.0

v0 = vmin + 20
##v0 = vmax - 20
##v0 = 0.5*(vmin+vmax)
w = 1.0

dv = 0.1
dxv = dv/vmax

v_lin = np.arange(vmin,vmax,dv)
N_log = int(np.ceil(np.log(vmax/vmin)/dxv))
v_log = vmin*np.exp(np.arange(N_log)*dxv)



print(v_lin[-1])
print(v_log[-1])


I_lin = gLv(v_lin,v0,w)
I_log = gLv(v_log,v0,w)

zp = 2
I_ditlin = spectrum(v_lin,
                    np.array([v0]),
                    np.array([np.log(w)]),
                    np.array([1.0]),
                    dxv,
                    folding_thresh = 1e-6,
                    zero_pad = zp
                    )

I_ditlog = spectrum(v_log,
                    np.array([v0]),
                    np.array([np.log(w)]),
                    np.array([1.0]),
                    dxv,
                    folding_thresh = 1e-6,
                    zero_pad = zp
                    )

##plt.plot(v_lin,I_ditlin,label= 'ditlin',lw=3)
##plt.plot(v_lin,I_lin,'--', label='lin',lw=3)
##plt.plot(v_log,I_ditlog, label = 'ditlog',lw=1)
##plt.plot(v_log,I_log, '--', label = 'log',lw=1)

##plt.plot(v_lin,I_ditlin - I_lin,label= 'ditlin - lin',lw=2)
plt.plot(v_log,I_ditlog - I_log, label = 'ditlog - log',lw=2)



plt.legend()
##plt.yscale('log')
plt.show()
