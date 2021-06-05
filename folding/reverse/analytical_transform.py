import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from matplotlib.patches import ConnectionPatch

gG = lambda v,w: (2/w)*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*(v/w)**2)
gL = lambda v,w: 2/(np.pi*w) / (1 + 4*(v/w)**2)

gG_FT = lambda x,w: np.exp(-(np.pi*x*w)**2/(4*np.log(2)))
gL_FT = lambda x,w: np.exp(- np.pi*x*w)

def gV(v,v0,wL,wG):
    gamma = (wG**5 + 2.69269*wG**4*wL + 2.42843*wG**3*wL**2 + 4.47163*wG**2*wL**3 + 0.07842*wG*wL**4 + wL**5)**0.2
    eta   = 1.36603*wL/gamma - 0.47719*(wL/gamma)**2 + 0.11116*(wG/gamma)**3
    return (1-eta) * gG(v,v0,gamma) + eta * gL(v,v0,gamma)

dv = 0.01 #cm-1
v_max = 100.0 #cm-1
v_arr = np.arange(0,v_max,dv)
N_zpad = v_arr.size ## can be 0..v_arr.size
##x_arr = np.arange(v_arr.size + 1) / (2 * v_arr.size * dv)
##x_arr = np.linspace(0,1/(2*dv),(v_arr.size + N_zpad)//2 + 1)
x_arr = np.fft.rfftfreq(v_arr.size + N_zpad, dv)
x_fold = (x_arr,x_arr[::-1])
folding_thresh = 1e-6

w_arr = np.array([5,2,1,0.05])*v_max

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.15)
ax1.axhline(0,c='k')
ax1.axvline(0,c='k')
ax2.axhline(0,c='k')
ax2.axvline(0,c='k')

for w in w_arr:

    I2 = gL(v_arr,w)
    I2_FT = np.fft.rfft(list(I2) + [0] + list(I2)[:0:-1])*dv
    p, = ax1.plot(v_arr,I2,label = '$w_G = {:.1f} \\times \\Delta\\nu$'.format(w/dv))
    c = p.get_color()
    ax2.plot(x_arr,I2_FT.real,c=c)
    
    I_FT = gL_FT(x_arr,w)
    I = np.fft.irfft(I_FT/dv)[:v_arr.size]
    

    c='k'
    ax1.plot(v_arr, I, '--', c=c, alpha=0.5)
    ax2.plot(x_arr, I_FT, '--', c=c, alpha=0.5)


N_ticks = 6
ax1.set_title('Real domain')
ax1.set_xlabel('$\\nu$ (cm$^{-1}$)')
ax1.set_xticks([i/(N_ticks-1)*v_max for i in range(N_ticks)])
ax1.set_xticklabels(['{:d}'.format(i)+' $\\cdot\\Delta\\nu$' for i in range(N_ticks)])
ax1.set_xlim(0,v_max)
ax1.legend(fontsize=11)    

N_ticks = 5
ax2.set_xlabel('$\\tilde{\\nu}$ (cm)')
ax2.set_xticks([i/(2*v_max) for i in range(N_ticks)])
ax2.set_xticklabels(['$\\frac{'+(str(Fraction(i,2*(N_ticks-1))) if i else '0/').replace('/','}{')+'\\Delta\\nu}$' for i in range(0,N_ticks)])
ax2.set_title('Fourier domain')
ax2.set_xlim(0,5/(2*v_max))

##con = ConnectionPatch(xyA=(1.07,0.47),
##                      xyB=(-0.07,0.47),
##                      coordsA="axes fraction",
##                      coordsB="axes fraction",
##                      axesA=ax1, axesB=ax2,
##                      arrowstyle="simple",fc='w')
##ax1.add_artist(con)
##ax1.indicate_inset_zoom(axins)
plt.savefig(__file__[:-2]+'png')
plt.show()
