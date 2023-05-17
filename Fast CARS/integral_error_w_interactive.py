import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import sys
from matplotlib.widgets import Slider, Button, RadioButtons

def gG(ksi):
    return 2*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*ksi**2)

def gL(ksi):
    return 2/((np.pi) * (4*ksi**2 + 1))

axcolor = 'lightgoldenrodyellow'
fig, (ax1,ax2) = plt.subplots(1,2)
v0w = 'v0'
dksi = 0.01
ksi_max = 4.0
ksi = np.arange(-ksi_max/2,ksi_max/2,dksi)

g = gG
x = 0.5
p = 1.2
a_arr = np.arange(0,1.0,0.001)

#The exact lineshape:
I    = g(ksi)

#The approximation:
Ia_0 = p** x   *g(ksi*p** x   )
Ia_1 = p**(x-1)*g(ksi*p**(x-1))

a1 = x
e1 = np.sum(((1-a1)*Ia_0 + a1*Ia_1 - I)**2)*dksi

a2 = (1-p**-x)/(1-p**-1) 
e2 = np.sum(((1-a2)*Ia_0 + a2*Ia_1 - I)**2)*dksi

a3 = 0.5*a1 + 0.5*a2
e3 = np.sum(((1-a3)*Ia_0 + a3*Ia_1 - I)**2)*dksi

a_curr = a3
e_curr = np.sum(((1-a3)*Ia_0 + a3*Ia_1 - I)**2)*dksi

Ia   = (1-a_curr)*Ia_0 + a_curr*Ia_1

#The error:
Ie   = Ia - I

ax1.plot(ksi,ksi-ksi,'gray')
p1, = ax1.plot(ksi,Ie**2)



e_list = []
for a in a_arr:
    Ia_0 = p** x   *g(ksi*p** x   )
    Ia_1 = p**(x-1)*g(ksi*p**(x-1))
    Ia   = (1-a)*Ia_0 + a*Ia_1
    e_RMS = np.sum((I - Ia)**2)*dksi
    e_list.append(e_RMS)

ax2.plot(a_arr,a_arr-a_arr,'gray')
p2, = ax2.plot(a_arr,e_list)
p3, = ax2.plot([a_curr],[e_curr],'o',label = 'Current $a$')
p4, = ax2.plot([a1],[e1],'x',label = '$a_1=x$')
p5, = ax2.plot([a2],[e2],'+',label = '$a_2=(1-p^{-x})/(1-p^{-1})$')
p6, = ax2.plot([a3],[e3],'.',label = '$a_3=(a_1+a_2)/2$')

ax1.grid(True)
ax1.set_title('Squared Approximation Error')
ax1.set_xlim(ksi[0],ksi[-1])
ax1.set_xlabel('$\\xi$')

ax2.set_title('Integral Square Error')
ax2.set_xlabel('$a$')
#ax2.set_ylabel('Error')
ax2.set_xlim(0,1)
ax2.grid()
ax2.legend(loc=1)

plt.subplots_adjust(bottom=0.3)

axx = plt.axes([0.28, 0.1,  0.6, 0.03], facecolor=axcolor)
axp = plt.axes([0.28, 0.15, 0.6, 0.03], facecolor=axcolor)
axa = plt.axes([0.28, 0.05,  0.6, 0.03], facecolor=axcolor)

sx  = Slider(axx, 'x', 0.0,  1.0, valinit=x)
sp  = Slider(axp, 'ln(p)', 1e-5, np.log(2), valinit=np.log(p))
sa  = Slider(axa, 'a', 0.0, 1.0, valinit=a3)

rax = plt.axes([0.1, 0.08, 0.10, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('G-w','L-w'), active=0)


def update_w(var):
    x = sx.val
    p = np.exp(sp.val)
    a = sa.val

    #The exact lineshape:
    I    = g(ksi)

    #The approximation:
    Ia_0 = p** x   *g(ksi*p** x   )
    Ia_1 = p**(x-1)*g(ksi*p**(x-1))
    Ia   = (1-a)*Ia_0 + a*Ia_1

    #The error:
    Ie = Ia - I

    p1.set_ydata(Ie**2)

    a_curr = a
    e_curr = np.sum(((1-a_curr)*Ia_0 + a_curr*Ia_1 - I)**2)*dksi

    a1 = x
    e1 = np.sum(((1-a1)*Ia_0 + a1*Ia_1 - I)**2)*dksi

    a2 = (1-p**-x)/(1-p**-1) 
    e2 = np.sum(((1-a2)*Ia_0 + a2*Ia_1 - I)**2)*dksi

    a3 = x - x*(x-1)/4*np.log(p) #0.5*a1 + 0.5*a2
    e3 = np.sum(((1-a3)*Ia_0 + a3*Ia_1 - I)**2)*dksi

    e_list = []
    for a in a_arr:
        Ia_0 = p** x   *g(ksi*p** x   )
        Ia_1 = p**(x-1)*g(ksi*p**(x-1))
        Ia   = (1-a)*Ia_0 + a*Ia_1
        e_RMS = np.sum((I - Ia)**2)*dksi
        e_list.append(e_RMS)
        
    p2.set_ydata(e_list)
    p3.set_xdata([a_curr])
    p3.set_ydata([e_curr])
    p4.set_xdata([a1])
    p4.set_ydata([e1])
    p5.set_xdata([a2])
    p5.set_ydata([e2])
    p6.set_xdata([a3])
    p6.set_ydata([e3])

    fig.canvas.draw_idle()


def change_g(label):
    global g,v0w
    l = label.split('-')
    g = {'G':gG,'L':gL}[l[0]]
    update_w(None)

sx.on_changed(update_w)
sp.on_changed(update_w)
sa.on_changed(update_w)
radio.on_clicked(change_g)
    
plt.show()

