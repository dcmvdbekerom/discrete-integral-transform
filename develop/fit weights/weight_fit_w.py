import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.special import comb
import sys
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.optimize import minimize, LinearConstraint

def gG(x):
    return 2*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*x**2)

def fitfun(a_list, t, n, I_ex):
    I_ap = np.zeros(len(x_arr))
    x0 = -(n//2 + t) * dx

    for k in range(n + 1):
        I_bin = g(x_arr - (x0 + k*dx))
##        plt.plot(x_arr,I_bin,'k',alpha=0.5)
        I_ap += a_list[k] * I_bin  

##    plt.plot(x_arr,I_ex,'k--')
    err = np.sum((I_ex - I_ap)**2)

    return err




x_max = 10.0
x_res = 0.05
x_arr = np.arange(-x_max, x_max, x_res)



dx = 0.5
n = 2 #order = number of bins - 1
g = gG
t = 0.25 #grid alignment




I_ex = g(x_arr)
##fitfun((n+1)*[1],t,n,I_ex)
dt = 0.05
t_arr = np.arange(0,1+dt,dt) - (1-n&1)/2

a_list = []

constr = LinearConstraint(np.ones(n+1),1,1)

for t in t_arr:
    res = minimize(fitfun,
                   x0=(n+1)*[1/(n+1)],
                   args=(t, n, I_ex),
                   method='trust-constr',
                   constraints=(constr,))
    a_list.append(res.x)
    
a_arr = np.array(a_list).T

for i in range(len(a_arr)):
    plt.plot(t_arr,a_arr[i],label=i)
    c = np.polyfit(t_arr,a_arr[i],n)
    print(c)
    plt.plot(t_arr,np.polyval(c,t_arr),'k--')
plt.legend()
plt.show()


