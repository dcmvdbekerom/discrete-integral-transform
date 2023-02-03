import numpy as np
from time import perf_counter
from add_module import numpy_add_at, cython_add_at
#from numba import jit
#from numba import vectorize, int32, float64

Nv = 300000
NG =  4
NL = 16
Ni = int(1e7)
##
##@vectorize
##def jit_add_at(arr,k,l,m,I):
##    shape = arr.shape
##    strides = 
##    lin_arr = arr.reshape((shape[0]*shape[1]*shape[2],))
##
##    for i in range(len(I)):
##        arr[k[i],l[i],m[i]] += I[i]
##    return arr

I = np.random.rand(Ni).astype(np.float32)
k = np.random.randint(Nv, size = Ni, dtype=np.int32)
l = np.random.randint(NG, size = Ni, dtype=np.int32)
m = np.random.randint(NL, size = Ni, dtype=np.int32)

LDM1 = np.zeros((Nv, NG, NL), dtype=np.float32)
t0 = perf_counter()
numpy_add_at(LDM1, k, l, m, I)
print(perf_counter() - t0)

LDM2 = np.zeros((Nv, NG, NL), dtype=np.float32)
t0 = perf_counter()
cython_add_at(LDM2, k, l, m, I)
print(perf_counter() - t0)

##print(np.sum(np.abs(LDM1-LDM2)))
