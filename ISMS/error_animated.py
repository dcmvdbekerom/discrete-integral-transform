import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import sys
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
from PIL import Image
mpl.rc('font',family='Times New Roman')

def gG(x):
    return 2*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*x**2)

def gL(x):
    return 2/((np.pi) * (4*x**2 + 1))

axcolor = 'lightgoldenrodyellow'
fig, ax1 = plt.subplots(1,1,figsize=(6.4,4.8))
v0w = 'w'
dx = 0.01
x_max = 4.0
x = np.arange(-x_max/2,x_max/2,dx)

cdict = {'red':  [(0.00,1,1),
                  (0.25,1,1),
                  (0.50,1,1),
                  (0.75,0,0),
                  (1.00,0,0)],

         'green':[(0.00,0,0),
                  (0.25,0,0),
                  (0.50,1,1),
                  (0.75,1,1),
                  (1.00,1,1)],
         
         'blue': [(0.00,1,1),
                  (0.25,0,0),
                  (0.50,0,0),
                  (0.75,0,0),
                  (1.00,1,1)],}

cmap = LinearSegmentedColormap('cmap',cdict)
colors = [cmap(i/4) for i in range(5)] 

clist = [ (0.00,0,0),
          (0.25,0.9,0.9),
          (0.50,0.9,0.9),
          (0.75,0.9,0.9),
          (1.00,0,0)]
cdict2 = {'red': clist,'green':clist,'blue':clist}         
cmap2 = LinearSegmentedColormap('cmap2',cdict2)

g = gL
Dx = 0.5
kt = 2.5

w = np.exp((kt-2)*Dx)
k = int(kt)
t = kt-k
a = t # Simple weight

w_arr = np.exp(np.arange(-2,3)*Dx)
Ie = g(x*w)*w
Im = g(x*w_arr[k  ])*w_arr[k  ]
Ip = g(x*w_arr[k+1])*w_arr[k+1]
I_list = [g(x*wi)*wi for wi in w_arr]

Ia = (1-a)*Im + a*Ip

dIdx   = np.ediff1d(Ie,to_end = 0)/dx
d2Idx2 = np.ediff1d(dIdx,to_end = 0)/dx
Y        = 2*x*dIdx + x**2*d2Idx2
Ierr_est = t*(1-t)*Dx**2/2 * Y

p_list = [ax1.plot(x,I,c='gray',lw=2)[0] for I in I_list]
pe, = ax1.plot(x,Ie,'--',lw=2,c='k',label= 'Exact Lineshape',zorder = 30)
pp, = ax1.plot(x,Ip,lw=4,c=colors[k+1],alpha = a,label='I[k+1]')
pabg, = ax1.plot(x,Ia,lw=8,c='w',zorder = 10)
pa, = ax1.plot(x,Ia,lw=4,c=cmap(kt/4),zorder = 20,label = 'Approximated Lineshape\n= {:.2f}$\\cdot$I[k] + {:.2f}$\\cdot$I[k+1]'.format(1-a,a))
pm, = ax1.plot(x,Im,lw=4,c=colors[k  ],alpha = 1-a,label='I[k]')


ax1.grid(True)
ax1.axhline(0,c='gray',zorder = -10)
ax1.axvline(0,c='gray',zorder = -10)
ax1.set_xlim(0,1.50)
ax1.set_xlabel('$\\nu$ (cm$^{-1}$)')
##ax1.set_ylim(0,np.exp( 2*Dx))
ax1.set_ylim(0,2.0)
ax1.legend(loc=1,fontsize=14)

plt.subplots_adjust(bottom=0.135,top=0.910)


def update(val):

    kt = val
    k = int(kt)
    t = kt - k
    a = t

    if v0w == 'v0':
        x = np.arange(-(x_max//(2*Dx))*Dx,(x_max//(2*Dx))*Dx+Dx,Dx)
        Ie = g(x - (kt-2)*Dx)
        Im = g(x - (k-2)*Dx)
        Ip = g(x - (k-1)*Dx)
        I_list = [g(x - (k-2)*Dx) for k in range(5)]
        
    else:
        x = np.arange(-x_max/2,x_max/2,dx)
        w     = np.exp((kt-2)*Dx)
        w_arr = np.exp(np.arange(-2,3)*Dx)

        Ie = g(x*w)*w
        Im = g(x*w_arr[k  ])*w_arr[k  ]
        Ip = g(x*w_arr[min((k+1,4))])*w_arr[min((k+1,4))]
        I_list = [g(x*wi)*wi for wi in w_arr]

        dIdx   = np.ediff1d(Ie,to_end = 0)/dx
        d2Idx2 = np.ediff1d(dIdx,to_end = 0)/dx
        Y        = 2*x*dIdx + x**2*d2Idx2
        Ierr_est = t*(1-t)*Dx**2/2 * Y


    Ia = (1-a)*Im + a*Ip

    pa.set_xdata(x)
    pa.set_ydata(Ia)
    pa.set_c(cmap(kt/4))
    pa.set_label('Approximated Lineshape\n= {:.2f}$\\cdot$I[k] + {:.2f}$\\cdot$I[k+1]'.format(1-a,a))

    pabg.set_xdata(x)
    pabg.set_ydata(Ia)

    
    [(p.set_xdata(x),p.set_ydata(I)) for I,p in zip(I_list,p_list)]

    pe.set_xdata(x)
    pe.set_ydata(Ie)

    pm.set_xdata(x)
    pm.set_ydata(Im)
    pm.set_alpha(1-a)
    pm.set_c(colors[k])

    pp.set_xdata(x)
    pp.set_ydata(Ip)
    pp.set_alpha(a)
    pp.set_c(colors[min(k+1,4)])

    ax1.legend(loc=1,fontsize=14)
    fig.canvas.draw_idle()

gif_images = []
for kt in np.linspace(0.0,4.0,150):
    print(kt)
    update(kt)
    fname = 'output/G_Dx={:.2f}_tk={:.3f}.png'.format(Dx,kt)
    plt.savefig(fname,dpi=100.0)
    pImage=Image.open(fname)
    gif_images.append(pImage.convert('RGB').convert('P', palette=Image.ADAPTIVE))

gif_images += gif_images[::-1]
gif_images[0].save('L_Dx={:.2f}.gif'.format(Dx),
                    save_all=True,
                    duration = 100,
                    loop = 0,
                    append_images=gif_images[1:])

