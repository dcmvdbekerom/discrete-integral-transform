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
##x_arr = np.arange(v_arr.size + 1) / (2 * v_arr.size * dv)
##x_arr = np.linspace(0,1/(2*dv),(v_arr.size + N_zpad)//2 + 1)
x_arr = np.fft.rfftfreq(v_arr.size + N_zpad, dv)
x_fold = (x_arr,x_arr[::-1])
folding_thresh = 1e-6

w_arr = np.array([5,2,1,0.5])*v_max

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
plt.subplots_adjust(left=0.05,right=0.95,bottom=0.15)
ax1.axhline(0,c='k')
ax1.axvline(0,c='k')
ax2.axhline(0,c='k')
ax2.axvline(0,c='k')

for w in w_arr:
    
    I_FT = gG_FT(x_arr,w)
    I = np.fft.irfft(I_FT/dv)[:v_arr.size]
    
    I2 = gG(v_arr,w)
    I2_FT = np.fft.rfft(list(I2) + [0] + list(I2)[:0:-1])*dv

##    I_FT *= I_FT[0]
##    I2_FT = np.fft.rfft(np.fft.fftshift(list(I-I2) + [0] + list(I-I2)[:0:-1]))*dv
##    I2_FT += I2_FT[-1]
##    I2_FT /= I2_FT[0]

##    I3 = gG_FT(x_arr,w/(4))
##    I3 /= I3[0]

    p, = ax1.plot(v_arr,I-I2,'--',alpha=0.5)
    c = p.get_color()
####    I_corr = -w/v_max * np.sin(2*np.pi*x_arr*v_max)/(2*np.pi*x_arr*v_max)
    ax2.plot(x_arr,I_FT,'--',c=c,alpha = 0.5)

   

##    ax1.plot(v_arr,I2,c=c,label = '$w_G = {:.1f} \\times \\Delta\\nu$'.format(w/dv))
    ax2.plot(x_arr,I2_FT.real,c=c)
##    ax2.set_yscale('log')
    
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
