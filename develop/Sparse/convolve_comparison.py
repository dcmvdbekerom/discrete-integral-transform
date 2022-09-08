import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import oaconvolve, fftconvolve, convolve
from numpy import pi
from time import perf_counter
from functools import partial

gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)

def sparse_convolve(S_arr, y_ls):
    S_out = np.zeros(S_arr.size + y_ls.size - 1)
    idx = np.arange(S_arr.size)[S_arr != 0.0]
    for i in idx:
        S_out[i:i+y_ls.size] += S_arr[i]*y_ls
    i0 = (y_ls.size + 1)//2 - 1
    return S_out[i0: i0 + S_arr.size]   


dv = 0.1
trunc = 5.0 #cm-1
v_ls = np.arange(-trunc,trunc,dv)
y_ls = gL(v_ls,0,1.0)

Nl = 2000
Nv = 20000000

Nv_arr = 2**np.arange(10,20)

f_dict = {'direct': partial(convolve, method='direct'),
          'fft': partial(convolve, method='fft'),
          'numpy':partial(np.convolve),
          'sparse':sparse_convolve,
          }

for key in f_dict:
    f = f_dict[key]
    t_list = []
    for Nv in Nv_arr:
        print(np.log2(Nv))
         
        S_ls = np.zeros(Nv)
        i0 = Nv//2-y_ls.size//2 
        S_ls[i0: i0 + y_ls.size] = y_ls

        t_out = 0
        for i in range(10):
            np.random.seed(i)
            S0_arr = np.random.rand(Nl)
            k0_arr = np.random.randint(0,Nv,Nl)
            S_arr = np.zeros(Nv)
            S_arr[k0_arr] = S0_arr

            t0 = perf_counter()
            res = f(S_arr,y_ls*dv)
            t_out += perf_counter() - t0

        t_list.append(t_out)

    plt.plot(Nv_arr, t_list, label=key)
    
plt.plot(Nv_arr,Nv_arr*1e-6,'k--')
plt.plot(Nv_arr,Nv_arr*(Nv_arr*1e-12),'k-.')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show()
