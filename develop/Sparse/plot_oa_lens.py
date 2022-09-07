from scipy.signal._signaltools import _calc_oa_lens as calc_oa_lens
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy import fft as sp_fft


def oa_lens(filter_len):
    overlap = filter_len - 1
    opt_size = np.ceil(-overlap*lambertw(-1/(2*np.e*overlap), k=-1).real).astype(int)
    block_size = np.array([sp_fft.next_fast_len(s) for s in opt_size])
    return block_size


n_arr = np.arange(10, 100000)
bs_arr = oa_lens(n_arr)
y_arr = -lambertw(-1/(2*np.e*n_arr), k=-1).real
plt.plot(n_arr,bs_arr/n_arr, '-')
plt.plot(n_arr,y_arr,'k-')
plt.xlabel('filter size')
plt.ylabel('block size / filter size')
plt.show()


