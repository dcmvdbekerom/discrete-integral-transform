import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.special import comb
import sys
from matplotlib.widgets import Slider, Button, RadioButtons

def gG(x):
    return 2*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*x**2)

def weight(t,n,k):
    return comb(n, k) * t**k * (1 - t)**(n-k)

##def approx(x_arr, t_a, n):
##    t = t_a/n + (n-1)/(2*n) #Bezier param
##
##    I_bins = []
##    x0_k = []
##    a_k = []
##
##    I_ap = np.zeros(len(x_arr))
##    
##    x0_0 = -t * n * dx
##    for k in range(n + 1):
##        x0_k.append(x0_0 + k*dx)
##        a_k.append(weight(t,n,k))
##        I_bins.append(g(x_arr - x0_k[-1]))
##        
##        I_ap += a_k[-1] * I_bins[-1]  
##
##    return I_ap, I_bins, x0_k, a_k

def approx(x_arr, t_a, n):
    t = t_a/n + (n-1)/(2*n) #Bezier param

    I_bins = []
    x0_k = []
    a_k = []

    I_bins.append(g(x_arr - t*dx))
    I_bins.append(g(x_arr - (t-1)*dx))
        
    I_ap = (1 - t_a) * I_bins[0] + t_a * I_bins[1]

    return I_ap, I_bins, x0_k, a_k




x_max = 10.0
x_res = 0.05
x_arr = np.arange(-x_max, x_max, x_res)



dx = 0.5
n = 1 #order = number of bins - 1
g = gG
t_a = 0.95 #grid alignment




I_ex = g(x_arr)
I_ap, I_bins, x0_k, a_k = approx(x_arr, t_a, n)

##I_a1, I_bins, x0_k, a_k = approx(x_arr, t_a, 1)
##I_a2, I_bins, x0_k, a_k = approx(x_arr, t_a, 2)

plt.bar(x0_k, a_k, width=dx*0.8)
for I in I_bins:
    plt.plot(x_arr, I, c='k', alpha=0.5)

plt.plot(x_arr, I_ap, c='r', lw=3)
##plt.plot(x_arr, I_a1, c='r')
##plt.plot(x_arr, I_a2, c='b')
plt.plot(x_arr, I_ex, 'k--')
plt.show()


