import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import sys
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')

def gG(x):
    return 2*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*x**2)

def gL(x):
    return 2/((np.pi) * (4*x**2 + 1))


fig,ax = plt.subplots(2)

kmin, kmax = 0, 30
lmin, lmax = 0, 6

vmin = 1800.0 #cm-1
wmin = 0.3 #cm-1

dv = 0.1 #cm-1
dxw = 0.3 #cm-1

vmax = vmin + (kmax - kmin)*dv
v_arr = np.arange(vmin,vmax,dv)


ax[0].set_xlim(kmin, kmax)
ax[0].set_ylim(lmin, lmax)
ax[0].set_xlabel('$k$')
ax[0].set_ylabel('$l$')
ax[0].set_title('$S[k,l]$:')
ax[0].set_aspect(1)
ax[0].grid(True)

ax[0].set_xticks(np.arange(kmin,kmax+1))
ax[0].set_yticks(np.arange(lmin,lmax+1))

colors = np.zeros((4,4), dtype=float)
colors[:,0] = 1.0
colors[:,3] = 1.0

pv, = ax[0].plot((0.5,0.5), (0,1),'k--', alpha=0.25, lw=1)
ph, = ax[0].plot((0,1), (0.5,0.5),'k--', alpha=0.25, lw=1)
psn, = ax[0].plot((0,0), (0,0), 'g-', alpha=0.25,lw=2, zorder=0)
pch, = ax[0].plot((0.5,), (0.5,),'k+', ms=9, zorder=20)
ppt = ax[0].scatter((0,0,1,1), (0,1,0,1), color=colors, marker='D', zorder=10)

ax[1].set_xlim(vmin, vmax)
ax[1].set_ylim(0, 1)
ax[1].set_xlabel('$\\nu$ (cm$^{-1}$)')
ax[1].set_ylabel('$I$ (a.u.)')
ax[1].set_title('$I(\\nu)$:')

ax[1].set_xticks(v_arr[::2])
ax[1].grid(True)

p00, = ax[1].plot(v_arr, wmin*gG(v_arr/wmin),'r', label='a00')
p01, = ax[1].plot(v_arr, wmin*gG(v_arr/wmin),'r', label='a01')
p10, = ax[1].plot(v_arr, wmin*gG(v_arr/wmin),'r', label='a10')
p11, = ax[1].plot(v_arr, wmin*gG(v_arr/wmin),'r', label='a11')
pa0, = ax[1].plot(v_arr, wmin*gG(v_arr/wmin),'b',lw=3, alpha=0.5, label='Approx.')
pa1, = ax[1].plot(v_arr, wmin*gG(v_arr/wmin),'w',lw=2, alpha=1)
pex, = ax[1].plot(v_arr, wmin*gG(v_arr/wmin),'k--',lw=1, label='Exact')
ax[1].legend(loc=1)

snapped = (-1, -1)

def update(event):
    if event.xdata and event.ydata:
        k = (event.xdata if snapped[0] < 0 else snapped[0])
        l = (event.ydata if snapped[1] < 0 else snapped[1])
        
        k0, l0 = int(k), int(l)
        tv, tw = k - k0, l - l0
        k1, l1 = k0 + 1, l0 + 1

        av = tv
        aw = tw

        a00 = (1 - av) * (1 - aw)
        a01 = (1 - av) * aw
        a10 = av * (1 - aw)
        a11 = av * aw

        pv.set_data((k,k),(l0,l1))
        ph.set_data((k0,k1),(l,l))
        pch.set_data((k,),(l,))
        ppt.set_offsets(np.c_[(k0,k0,k1,k1),(l0,l1,l0,l1)])
        colors[:,3] = (a00,a01,a10,a11)
        ppt.set_color(colors)

        v0  = vmin + k0*dv
        v1  = vmin + k1*dv
        vex = vmin + k *dv
        w0  = wmin * np.exp(l0 * dxw)
        w1  = wmin * np.exp(l1 * dxw)
        wex = wmin * np.exp(l  * dxw)

        A = 0.5
        g = gL
        y00 = A * g((v_arr - v0 ) / w0 ) / w0
        y01 = A * g((v_arr - v0 ) / w1 ) / w1
        y10 = A * g((v_arr - v1 ) / w0 ) / w0
        y11 = A * g((v_arr - v1 ) / w1 ) / w1
        yex = A * g((v_arr - vex) / wex) / wex
        yap = a00*y00 + a01*y01 + a10*y10 + a11*y11

        ph.set_alpha((0.0 if not tv else 0.25))
        pv.set_alpha((0.0 if not tw else 0.25))
        
        p00.set_ydata(y00)
        p01.set_ydata(y01)
        p10.set_ydata(y10)
        p11.set_ydata(y11)
        pa0.set_ydata(yap)
        pa1.set_ydata(yex)
        pex.set_ydata(yex)

        p00.set_alpha(a00)
        p01.set_alpha(a01)
        p10.set_alpha(a10)
        p11.set_alpha(a11)
        
        ax[1].legend(loc=1)
        
        fig.canvas.draw_idle()
        

def on_press(event):
    global snapped
    if event.key == 'shift':
        if event.xdata and event.ydata:
            k, l = event.xdata, event.ydata
            ki, li = int(k + 0.5), int(l + 0.5)
            if np.abs(ki - k) < np.abs(li - l):
                snapped = (ki, -1)
                psn.set_data((ki, ki), (0, lmax))
            else:
                snapped = (-1, li)
                psn.set_data((0, kmax), (li, li))
            update(event)

        
def on_release(event):
    global snapped
    if event.key == 'shift':
        snapped = (-1, -1)
        psn.set_data((0,0), (0,0))
        update(event)

fig.canvas.mpl_connect('motion_notify_event', update)
fig.canvas.mpl_connect('key_press_event', on_press)
fig.canvas.mpl_connect('key_release_event', on_release)

plt.show()
