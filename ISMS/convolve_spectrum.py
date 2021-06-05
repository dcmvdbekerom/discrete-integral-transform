from radis.lbl.broadening import olivero_1977, voigt_lineshape
import matplotlib.pyplot as plt
import numpy as np

S0i,v0i,wLi,wGi = np.load('CO_db.npy')
S0i /= 3e-19
dv = 0.002 #cm-1
v_arr = np.arange(2050, 2220, dv)
I_arr = np.zeros(v_arr.size)
I_list = []
v_list = []

fig,ax = plt.subplots(2)
ax[0].set_ylim(-0.1,1)
ax[1].set_ylim(-0.1,1)
ax[0].axes.yaxis.set_visible(False)
ax[1].axes.yaxis.set_visible(False)
ax[0].axhline(0,c='k',alpha=0.5)
ax[1].axhline(0,c='k',alpha=0.5)
plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.95)

plt.xlabel('$\\nu$ (cm$^{-1}$)',fontsize=16)

v_arr_ls = v_arr - np.mean(v_arr)
wG = wGi[10]
wL = wLi[10]
wV = olivero_1977(wG,wL)
I = 1.5*voigt_lineshape(v_arr_ls, wL/2, wV/2, jit=False)

ax[0].plot([0,0],[0,10],'-',lw=1,c='k',alpha=0.25)
p1, = ax[0].plot(v_arr_ls,I)

lines = ax[1].vlines(v0i,0,S0i*0.6)

plt.savefig('convolution_before.png',dpi=100)
    
for S0, v0 in zip(S0i, v0i):
    I = S0 * voigt_lineshape(v_arr - v0, wL/2, wV/2, jit=False)
    I_arr += I

p1.remove()
ax[0].plot(v_arr_ls,np.zeros(v_arr.size),c='tab:blue')
ax[1].plot(v_arr,I_arr)
lines.remove()
plt.savefig('convolution_after.png',dpi=100)
plt.show()

