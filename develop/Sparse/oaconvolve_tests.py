import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import oaconvolve, fftconvolve, convolve
from scipy.signal._signaltools import _calc_oa_lens as calc_oa_lens
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
    return S_out[i0:i0 + S_arr.size]   


def my_oaconvolve(S_arr, ls, method='auto', mode='same'):
    block_size, overlap, in1_step, in2_step = calc_oa_lens(S_arr.size, ls.size)
    N = int(np.ceil(S_arr.size/in1_step))
    output = np.zeros(N*block_size)

    ls_pad = np.append(ls, np.zeros(block_size - ls.size))
    ls_FT = np.fft.rfft(ls_pad)
    
    for i in range(N):
        i0 = i*in1_step
        i1 = np.min((i0 + block_size - overlap, S_arr.size))
        block = S_arr[i0:i1]
        if block.size != block_size - overlap:
            block = np.append(block, np.zeros(block_size - overlap - block.size))
        block_pad = np.append(block, np.zeros(overlap))
        block_FT = np.fft.rfft(block_pad)
        
        output[i0:i0+block_size] += np.fft.irfft(ls_FT * block_FT).real
##        output[i0:i1+overlap] += convolve(block, ls, method=method, mode='full')
        
    i0 = (ls.size - 1)//2
    i1 = i0 + S_arr.size
    return output[i0:i1]



dv = 0.1
trunc = 5.0 #cm-1

Nls = 1000
v_ls = np.arange(-Nls,Nls+1)*dv
y_ls = gL(v_ls,0,1.0)


Nl = 200
Nv = 200000


block_size, overlap, in1_step, in2_step = calc_oa_lens(Nv, y_ls.size)
print(y_ls.size, block_size, Nv)
print(block_size/y_ls.size)

np.random.seed(0)
S0_arr = np.random.rand(Nl)
k0_arr = np.random.randint(0,Nv,Nl)
S_arr = np.zeros(Nv)
S_arr[k0_arr] = S0_arr


f_dict = {#'direct': partial(convolve, mode='same', method='direct'),
          #'fft': partial(convolve, mode='same', method='fft'),
          'oa': partial(oaconvolve, mode='same'),
          'my_oa_fft': partial(my_oaconvolve, mode='same', method='fft'),
          #'my_oa_dir': partial(my_oaconvolve, mode='same', method='direct'),
          'sparse':sparse_convolve,
          }


for i, key in enumerate(f_dict.keys()):
    f = f_dict[key]
    t0 = perf_counter()
    Sc = f(S_arr, y_ls*dv)
    print('{:20s}: {:.6f}'.format(key, perf_counter()-t0))
    
    plt.plot(Sc, lw = 1+2*i, zorder = -i, label=key)
    
plt.legend(loc=0)
plt.show()
