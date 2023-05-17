import numpy as np
import matplotlib.pyplot as plt

gG_FT = lambda x: np.exp(-(np.pi*x)**2/(4*np.log(2)))
gL_FT = lambda x: np.exp(-np.pi*x)

C1 = 1.06920
C2 = 0.86639

aG = lambda d: 2*(1-d)/(C1*(1+d)+(C2*(1+d)**2+4*(1-d)**2)**0.5)
aL = lambda d: 2*(1+d)/(C1*(1+d)+(C2*(1+d)**2+4*(1-d)**2)**0.5)

gV_FT = lambda x,d: gG_FT(x*aG(d))*gL_FT(x*aL(d))/dv

def gV(v,d):
    dv = (v[-1]-v[0])/(v.size-1)
    x = np.arange(v.size + 1) / (2 * v.size * dv)
    I = np.fft.ifftshift(np.fft.irfft(gV_FT(x,d)))[v.size//2:-v.size//2]
    return I/np.max(I)

v_max = 10.0
dv = 0.01
v = np.arange(-v_max,v_max,dv)

p1, = plt.plot(v,gV(v,0.0),'-')
plt.axhline( 0.0,c='k')
plt.axhline( 0.5,c='k')
plt.axvline( 0.5,c='k')
plt.axvline(-0.5,c='k')
plt.xlim(-2.5,2.5)

def on_plot_hover(event):
    global var
    var = event
    if event.xdata:
        d = np.clip(event.xdata/2.5,-1,1)
        y = gV(v,d)
        p1.set_ydata(y)
        plt.gcf().canvas.draw_idle()

plt.gcf().canvas.mpl_connect('motion_notify_event', on_plot_hover)   





plt.show()
