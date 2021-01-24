import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

gG = lambda v,w: (2/w)*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*(v/w)**2)
gL = lambda v,w: (1/np.pi) * (w/2) / (v**2 + (w/2)**2)*dv

gG_FT = lambda x,w: np.exp(-(np.pi*x*wG)**2/(4*np.log(2)))
gL_FT = lambda x,w: np.exp(- np.pi*x*wL)

dv = 0.001 #cm-1
v_max = 100.0 #cm-1
v_arr = np.arange(0,v_max,dv)
N_zpad = v_arr.size ## can be 0..v_arr.size
x_arr = np.linspace(-1/dv,1/dv,v_arr.size*4)
folding_thresh = 1e-6

wG = dv

# Current RADIS implementation:

fig, (ax2) = plt.subplots(1,1,figsize=(10,4))
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.15)


plt.plot(x_arr,gG_FT(x_arr,wG))
for n in range(1,3):
    plt.plot(x_arr,gG_FT( n/dv + x_arr,wG))
    plt.plot(x_arr,gG_FT(-n/dv + x_arr,wG))


ax2.axvspan(-1/dv, 0, alpha=0.5, color='w',zorder = 10)
ax2.axvspan(1/(2*dv), 1/dv, alpha=0.5, color='w',zorder = 10)

ax2.axhline(0,c='k',zorder = 20)
ax2.axvline(0,c='k',zorder = 20)
ax2.axvline(1/(2*dv),c='k',ls='--',zorder = 20)

N_ticks = 8

ax2.set_xlabel('$\\tilde{\\nu}$ (cm)')
ticks = [i/(N_ticks*dv) for i in range(-N_ticks,N_ticks+1)]
labels = ['$\\frac{'+(str(Fraction(i,N_ticks)) if i else '0/').replace('/','}{')+'\\Delta\\nu}$' for i in range(1-N_ticks,N_ticks)]
labels = ['$\\frac{-1}{\\Delta\\nu}$'] + labels + ['$\\frac{1}{\\Delta\\nu}$'] 
ax2.set_xticks(ticks)
ax2.set_xticklabels(labels)
ax2.set_title('Fourier domain')
ax2.set_xlim(-1/dv,1/dv)

plt.savefig(__file__[:-2]+'png')
plt.show()
