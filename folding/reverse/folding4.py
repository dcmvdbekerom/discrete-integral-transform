import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from matplotlib.patches import ConnectionPatch

gG = lambda v,w: (2/w)*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*(v/w)**2)
gL = lambda v,w: (1/np.pi) * (w/2) / (v**2 + (w/2)**2)*dv

gG_FT = lambda x,w: np.exp(-(np.pi*x*w)**2/(4*np.log(2)))
gL_FT = lambda x,w: np.exp(- np.pi*x*w)

dv = 0.01 #cm-1
v_max = 100.0 #cm-1
v_arr = np.arange(0,v_max,dv)
N_zpad = v_arr.size ## can be 0..v_arr.size
x_arr = np.fft.rfftfreq(v_arr.size + N_zpad, dv)
x_fold = (x_arr,x_arr[::-1])
folding_thresh = 1e-6

wG = v_max

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.15)

ax1.axhline(0,c='k')
ax1.axvline(0,c='k')
ax2.axhline(0,c='k')
ax2.axvline(0,c='k')

I_FT = gG_FT(x_arr,wG)
I = np.fft.irfft(I_FT/dv)[:v_arr.size]

I2 = gG(v_arr,wG)
I2_FT = np.fft.rfft(list(I2) + [0] + list(I2)[:0:-1])*dv # this assumes N_zpad = v_arr.size

ax1.plot(v_arr,I2,c='k',label = 'Direct'.format(wG/dv))
ax1.plot(v_arr,I,'--',label='FT - 1 fold')

ax2.plot(x_arr,I2_FT.real,c='k') ## I2 is symmetric, so I2_FT has only real components
ax2.plot(x_arr,I_FT,'--')

# Apply folding if needed: 
for n in range(1,3):
    I_FT += gG_FT(n/(2*dv) + x_fold[n&1],wG)
    I = np.fft.irfft(I_FT/dv)[:v_arr.size]

    ax1.plot(v_arr,I,'--',label='FT - {:d} folds'.format(n+1))
    ax2.plot(x_arr,I_FT,'--')


N_ticks = 6
ax1.set_title('Real domain')
ax1.set_xlabel('$\\nu$ (cm$^{-1}$)')
ax1.set_xticks([i*v_max/(N_ticks-1) for i in range(N_ticks)])
ax1.set_xticklabels(['{:d}'.format(i)+' $\\cdot\\Delta\\nu$' for i in range(N_ticks)])
ax1.set_xlim(0,v_max)
ax1.legend(fontsize=11)    

N_ticks = 5
ax2.set_xlabel('$\\tilde{\\nu}$ (cm)')
ax2.set_xticks([i/(2*v_max/(N_ticks-1)) for i in range(N_ticks)])
ax2.set_xticklabels(['$\\frac{'+(str(Fraction(i,2*(N_ticks-1))) if i else '0/').replace('/','}{')+'\\Delta\\nu}$' for i in range(0,N_ticks)])
ax2.set_title('Fourier domain')
ax2.set_xlim(0,1/(2*v_max/(N_ticks-1)))

plt.savefig(__file__[:-2]+'png')
plt.show()
