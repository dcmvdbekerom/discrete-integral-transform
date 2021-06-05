from radis.lbl.broadening import olivero_1977, voigt_lineshape
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

S0i,v0i,wLi,wGi = np.load('CO_db.npy')
S0i /= 3e-19
dv = 0.002 #cm-1
v_arr = np.arange(2050, 2220, dv)
I_arr = np.zeros(v_arr.size)


plt.ion()
fig,ax = plt.subplots(2,sharex=True)
p1, = ax[0].plot(v_arr,I_arr)
p2, = ax[1].plot(v_arr,I_arr)
ax[0].set_ylim(-0.1,1)
ax[1].set_ylim(-0.1,1)
ax[0].axes.yaxis.set_visible(False)
ax[1].axes.yaxis.set_visible(False)
ax[0].axhline(0,c='k',alpha=0.5)
ax[1].axhline(0,c='k',alpha=0.5)
plt.subplots_adjust(left = 0.05, right = 0.95, top = 0.95)

plt.xlabel('$\\nu$ (cm$^{-1}$)',fontsize=16)

i = 0

fname1 = 'output/add1_{:d}.png'
fname2 = 'output/add2_{:d}.png'

gif_images = []
for S0, v0, wL, wG in zip(S0i, v0i, wLi, wGi):

    wV = olivero_1977(wG, wL)
    I = S0 * voigt_lineshape(v_arr - v0, wL/2, wV/2, jit=False)

    p1.set_ydata(I)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.savefig(fname1.format(i),dpi=100.0)
    pImage=Image.open(fname1.format(i))
    gif_images.append(pImage.convert('RGB').convert('P', palette=Image.ADAPTIVE))

    I_arr += I

    p1.set_ydata(np.zeros(v_arr.size))
    p2.set_ydata(I_arr)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.savefig(fname2.format(i),dpi=100.0)
    pImage=Image.open(fname2.format(i))
    gif_images.append(pImage.convert('RGB').convert('P', palette=Image.ADAPTIVE))

    print(i)
    i+=1
    
gif_images[0].save('build_spectra.gif',
                    save_all=True,
                    duration = i*[500,100],
                    loop = 0,
                    append_images=gif_images[1:])
