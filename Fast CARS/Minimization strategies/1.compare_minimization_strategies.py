import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import sys
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.optimize import minimize
from numpy import exp, log, pi

## This version has been tested and is confirmed to be consistent with the paper (up to eq A28)


def GG(x):
    return 2*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*x**2)

def GL(x):
    return 2/((np.pi) * (4*x**2 + 1))

def GG_FT(x):
    return np.exp(-(np.pi*x)**2/(4*np.log(2)))

def GL_FT(x):
    return np.exp(-np.pi*np.abs(x))

dx = 0.001
x_max = 10.0
x = np.arange(-x_max/2,x_max/2,dx)
Dx = 0.2
Nx = x.size

##p = 1.2

G_dict = {'G':[GG,GG_FT],
          'L':[GL,GL_FT]}

G = GG
G_FT = GG_FT

t = 0.5
a_list=[
t,
(1 - exp(-t*Dx))/(1 - exp(-Dx)),
t + t*(1-t)*Dx/2 * 0.5
]

a_list.append(0.5*a_list[0] + 0.5*a_list[1]),

fig, ax = plt.subplots(1,2, sharex=True, sharey=True)


I = G(x)
I_FT = G_FT(x)

labels = ['Simple', 'Min. Peak', 'Min-RMS', 'Mean']

w_w0 = exp( t   *Dx)
w_w1 = exp((t-1)*Dx)

I0 = G(x*w_w0) * w_w0
I1 = G(x*w_w1) * w_w1

I0_FT = G_FT(x/w_w0)
I1_FT = G_FT(x/w_w1)


xdIdx   =  x[2:-2]    * (I_FT[3:-1] -   I_FT[1:-3]              )/(2*dx   )
x2d2Idx2 = x[2:-2]**2 * (I_FT[1:-3] - 2*I_FT[2:-2] +  I_FT[3:-1])/   dx**2


h1 = -pi*np.abs(x)
h2 =  pi*np.abs(x)*(pi*np.abs(x) - 1)

h1 = -pi**2*x**2/(2*log(2))
h2 =  pi**2*x**2/(2*log(2))*(pi**2*x**2/(2*log(2)) - 2)


for label, a in zip(labels,a_list):

    Ia    = (1-a)*I0    + a*I1
    DS    = (I    - Ia   )**2

    c1 = (a - t)*Dx
    c2 = 0.5*(t**2 -2*a*t +a)*Dx**2

##    err = (c1*xdIdx + c2*(xdIdx + x2d2Idx2))
##    err = (c1*h1 + c2*h2)*I_FT
    err_sq = (c1**2*h1**2 + 2*c1*c2*h1*h2 + c2**2*h2*h2)*I_FT**2

    SL1L1 = np.sum(h1**2*I_FT**2)*dx
    SL1L2 = np.sum(h1*h2*I_FT**2)*dx
    SL2L2 = np.sum(h2**2*I_FT**2)*dx

        
    Ia_FT = (1-a)*I0_FT + a*I1_FT
    DS_FT = (I_FT - Ia_FT)**2

    RMS    = (np.sum(DS)*dx)**0.5
    RMS_FT = (np.sum(DS_FT)*dx)**0.5

    print(RMS, RMS_FT)

    ax[0].plot(x,DS,    label=label)
    ax[1].plot(x,DS_FT, label=label)
    ax[1].plot(x,err_sq,'k--')


##dgdx   = np.ediff1d(g(x),to_end = 0)/dx
##d2gdx2 = np.ediff1d(dgdx,to_end = 0)/dx
##d2gdx2[-2:] = 0.0
##
##
##
##err1 = 0.50*t*(1-t)*np.log(p)**2 * (g(x) + 3*x*dgdx +   x**2*d2gdx2) 
##plt.plot(x,err1**2,'k--')
##
##err2 = 0.50*t*(1-t)*np.log(p)**2 * (       2*x*dgdx +   x**2*d2gdx2)
##plt.plot(x,err2**2,'k--')
##
##err3 = 0.25*t*(1-t)*np.log(p)**2 * (g(x) + 5*x*dgdx + 2*x**2*d2gdx2)
##plt.plot(x,err3**2,'k--')

plt.legend()
plt.show()
