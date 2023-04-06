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

axcolor = 'lightgoldenrodyellow'
fig, (ax1,ax2) = plt.subplots(1,2,sharex=True)
v0w = 'w'
dx = 0.02
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
k0 = int(kt)
k1 = k0 + 1
t = kt - k0

a0 = 1-t
a1 = t # Simple weight

##Ie = g(x + (kt-2)*Dx)
##Im = g(x + (k-2)*Dx)
##Ip = g(x + (k-1)*Dx)
##I_list = [g(x + (k-2)*Dx) for k in range(5)]

w_arr = np.exp(np.arange(-2,3)*Dx)
Ie = g(x*w)*w
I0 = g(x*w_arr[k0])*w_arr[k0]
I1 = g(x*w_arr[k1])*w_arr[k1]
I_list = [g(x*wi)*wi for wi in w_arr]

Ia = a0*I0 + a1*I1

xdIdx = x[1:-1] * (Ie[2:] - Ie[:-2])/(2*dx)
x2d2Idx2 = x[1:-1]**2 * (Ie[2:] - 2*Ie[1:-1] + Ie[:-2])/dx**2
Y        = Ie[1:-1] + 3*xdIdx + x2d2Idx2
Ierr_est = t*(1-t)*Dx**2/2 * Y
print('Max Y2:', np.max(np.abs(Y))) #0.740


p_list = [ax1.plot(x,I,c='gray',lw=2)[0] for I in I_list]
pe, = ax1.plot(x,Ie,'--',lw=2,c='k',label= 'Exact Lineshape',zorder = 30)
pabg, = ax1.plot(x,Ia,lw=6,c='w',zorder = 10)
pa, = ax1.plot(x,Ia,lw=4,c=cmap(kt/4),zorder = 20,label = 'Approximated Lineshape\n= {:.2f}$\\cdot$I[k] + {:.2f}$\\cdot$I[k+1]'.format(a0,a1))

p0, = ax1.plot(x,I0,lw=4,c=colors[k0],alpha = a0,label='I[k]')
p1, = ax1.plot(x,I1,lw=4,c=colors[k1],alpha = a1,label='I[k+1]')

p_err, = ax2.plot(x,Ia-Ie,lw = 4,c=cmap(kt/4),label='Approximation Error\n= I$_{approx.}$ - I$_{exact}$')
p_err_e, = ax2.plot(x[1:-1],Ierr_est,'k--',lw = 2,zorder = 20,label='Estimated error')



ax1.grid(True)
ax1.axhline(0,c='gray',zorder = -10)
ax1.axvline(0,c='gray',zorder = -10)
ax1.set_title('Approximation')
ax1.set_xlim(x[0],x[-1])
ax1.set_xlabel('$x_\\nu$')
ax1.set_ylim(0,np.exp( 2*Dx))
ax1.legend(loc=1,fontsize=14)

ax2.grid(True)
ax2.axhline(0,c='gray',zorder = -10)
ax2.axvline(0,c='gray',zorder = -10)
ax2.legend(loc=1,fontsize=14)

plt.subplots_adjust(bottom=0.3)

axx = plt.axes([0.28, 0.1,  0.6, 0.03], facecolor=axcolor)
axp = plt.axes([0.28, 0.15, 0.6, 0.03], facecolor=axcolor)

sp  = Slider(axp, '$\\Delta x$', 0, 1, valinit=Dx)
sx  = Slider(axx, 'k+t', 0.5,  3.49, valinit=kt)

rax = plt.axes([0.1, 0.08, 0.10, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('G-v0','G-w','L-v0','L-w'), active=3)
plt.ion()


def update(val):

    Dx = sp.val
    kt = sx.val

    
    k0 = int(kt)
    k1 = k0 + 1
    t = kt - k0
##    a0 = 1-t
##    a1 = t

    a1 = (np.exp(-t*Dx) - 1)/(np.exp(-Dx) - 1)
    a0 = 1 - a1
##    print(a1, t)
    
    if v0w == 'v0':
        x = np.arange(-(x_max//(2*Dx))*Dx,(x_max//(2*Dx))*Dx+Dx,Dx)
        Ie = g(x - (kt-2)*Dx)
        Im = g(x - (k-2)*Dx)
        Ip = g(x - (k-1)*Dx)
        I_list = [g(x - (k-2)*Dx) for k in range(3)]

##        dIdx   = np.ediff1d(Ie,to_end = 0)/dx
##        d2Idx2 = np.ediff1d(dIdx,to_end = 0)/dx
##        Ierr_est   = t*(1-t)*Dx**2/2 * d2Idx2 
        
    else:
        x = np.arange(-x_max/2,x_max/2,dx)
        w     = np.exp((kt-2)*Dx)
        w_arr = np.exp(np.arange(-2,3)*Dx)

        Ie = g(x*w)/w
        I0 = g(x*w_arr[k0])/w_arr[k0]
        I1 = g(x*w_arr[min((k1,4))])/w_arr[min((k1,4))]
        I_list = [g(x*wi)*wi for wi in w_arr]

        xdIdx = x[1:-1] * (Ie[2:] - Ie[:-2])/(2*dx)
        x2d2Idx2 = x[1:-1]**2 * (Ie[2:] - 2*Ie[1:-1] + Ie[:-2])/dx**2
        Y = Ie[1:-1] + 3*xdIdx + x2d2Idx2
        Ierr_est = t*(1-t)*Dx**2/2 * Y

    Ia = a0*I0 + a1*I1

    pa.set_xdata(x)
    pa.set_ydata(Ia)
    pa.set_c(cmap(kt/4))
    pa.set_label('Approximated Lineshape\n= {:.2f}$\\cdot$I[k] + {:.2f}$\\cdot$I[k+1]'.format(a0,a1))

    pabg.set_xdata(x)
    pabg.set_ydata(Ia)

    
    [(p.set_xdata(x),p.set_ydata(I)) for I,p in zip(I_list,p_list)]

    pe.set_xdata(x)
    pe.set_ydata(Ie)

    p0.set_xdata(x)
    p0.set_ydata(I0)
##    p0.set_alpha(a0)
    p0.set_c(colors[k0])

    p1.set_xdata(x)
    p1.set_ydata(I1)
##    p1.set_alpha(a1)
    p1.set_c(colors[min(k1,4)])

    p_err.set_xdata(x)
    p_err.set_ydata(Ia-Ie)
    p_err.set_c(cmap(kt/4))
    p_err_e.set_ydata(Ierr_est)

    ax1.legend(loc=1,fontsize=14)
    ax2.legend(loc=1,fontsize=14)
    fig.canvas.draw_idle()


def change_g(label):
    global g,v0w
    label = label.split('-')
    g   = {'G':gG,'L':gL}[label[0]]
    v0w = label[1]
    update(None)

sx.on_changed(update)
sp.on_changed(update)
radio.on_clicked(change_g)
    
plt.show()

