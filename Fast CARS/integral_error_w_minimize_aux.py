import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import sys
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.optimize import minimize

def gG(x):
    return 2*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*x**2)

def gL(x):
    return 2/((np.pi) * (4*x**2 + 1))

def gG_FT(x):
    return np.exp(-(np.pi*x)**2/(4*np.log(2)))

def gL_FT(x):
    return np.exp(-np.pi*np.abs(x))

dx = 0.001
x_max = 10.0
x = np.arange(-x_max/2,x_max/2,dx)
p = 1.2

g = gL_FT
g = gL
t = 0.5
a1 = t
a2 = (1 - p**-t)/(1 - p**-1)
##a3 = t + t*(t-1)*np.log(p)/2
a3 = 0.5*a1 + 0.5*a2

I = g(x)

for i in range(3):

    a = [a1,a2,a3][i]
    Ia_0 = g(x*p** t   ) * p**t
    Ia_1 = g(x*p**(t-1)) * p**(t-1)
    Ia   = (1-a)*Ia_0 + a*Ia_1

    plt.plot(x,(I-Ia)**2,label=i)

dgdx   = np.ediff1d(g(x),to_end = 0)/dx
d2gdx2 = np.ediff1d(dgdx,to_end = 0)/dx
d2gdx2[-2:] = 0.0



err1 = 0.50*t*(1-t)*np.log(p)**2 * (g(x) + 3*x*dgdx +   x**2*d2gdx2) 
plt.plot(x,err1**2,'k--')

err2 = 0.50*t*(1-t)*np.log(p)**2 * (       2*x*dgdx +   x**2*d2gdx2)
plt.plot(x,err2**2,'k--')

err3 = 0.25*t*(1-t)*np.log(p)**2 * (g(x) + 5*x*dgdx + 2*x**2*d2gdx2)
plt.plot(x,err3**2,'k--')

plt.legend()
plt.show()
