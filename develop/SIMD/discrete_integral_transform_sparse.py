## Simple implementation with core functionality but no error corrections

import numpy as np
from time import perf_counter
import pyfftw
from scipy.fft import rfft, irfft

fft_fwd = None
fft_rev = None

def aligned_zeros(shape, alignment=32, dtype=np.float32, order='c', **kwargs):
    N = np.prod(shape)
    dsize = np.dtype(dtype).itemsize
    arr = np.zeros(N + alignment // dsize, dtype=dtype, **kwargs)
    diff = arr.ctypes.data % alignment
    start = diff // dsize
    return arr[start:start + N].reshape(shape, order=order)


def init_w_axis(dx, log_wi):
    log_w_min = np.min(log_wi)
    log_w_max = np.max(log_wi) + 1e-4
    N = np.ceil((log_w_max - log_w_min)/dx) + 1
    log_w_arr = log_w_min + dx * np.arange(N)
    return log_w_arr.astype(np.float32)
 
    
def get_indices(arr_i, axis):
    pos   = np.interp(arr_i, axis, np.arange(axis.size))
    index = pos.astype(int)
    a = pos - index
    return index, index + 1, a


def get_indices2(arr_i, axis, dx):
    Nx = axis.size
    x_min = axis[0]
##    dx = axis[1] - axis[0]
    print(dx)
    pos   = (arr_i - x_min) / dx
    index = pos.astype(int)
    a = pos - index
    return index, index + 1, a


## Calc matrix functions:

def calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i):
    
    S_klm = np.zeros((2 * v.size, log_wG.size, log_wL.size))
    
    ki0, ki1, avi = get_indices2(v0i, v, 0.002)          #Eqs 3.4 & 3.6
    li0, li1, aGi = get_indices2(log_wGi, log_wG, 0.1) #Eqs 3.7 & 3.10
    mi0, mi1, aLi = get_indices2(log_wLi, log_wL, 0.4) #Eqs 3.7 & 3.10
    
    np.add.at(S_klm, (ki0, li0, mi0), S0i * (1-avi) * (1-aGi) * (1-aLi))
    np.add.at(S_klm, (ki0, li0, mi1), S0i * (1-avi) * (1-aGi) *    aLi )
    np.add.at(S_klm, (ki0, li1, mi0), S0i * (1-avi) *    aGi  * (1-aLi))
    np.add.at(S_klm, (ki0, li1, mi1), S0i * (1-avi) *    aGi  *    aLi )
    np.add.at(S_klm, (ki1, li0, mi0), S0i *    avi  * (1-aGi) * (1-aLi))
    np.add.at(S_klm, (ki1, li0, mi1), S0i *    avi  * (1-aGi) *    aLi )
    np.add.at(S_klm, (ki1, li1, mi0), S0i *    avi  *    aGi  * (1-aLi))
    np.add.at(S_klm, (ki1, li1, mi1), S0i *    avi  *    aGi  *    aLi )
    
    return S_klm



## Apply_transform functions:

def apply_transform(v, log_wG, log_wL, S_klm):
    dv     = v[1] - v[0]
    x      = np.fft.rfftfreq(S_klm.shape[0], dv)
    S_k_FT = np.zeros(x.size, dtype = np.complex64)
    
    for l in range(log_wG.size):
        for m in range(log_wL.size):
            wG_l,wL_m = np.exp(log_wG[l]),np.exp(log_wL[m])
            gV_FT = np.exp(-((np.pi*wG_l*x)**2/(4*np.log(2)) + np.pi*wL_m*x))
            S_k_FT += np.fft.rfft(S_klm[:,l,m]) * gV_FT

    print('(FT:~~~~, MM:~~~~) - ',end='')
            
    return np.fft.irfft(S_k_FT)[:v.size] / dv    


## Synthesizie_spectrum function:

def plan_FFTW(Nv, NwG, NwL, wisdom_file='wisdom.txt', patience='FFTW_ESTIMATE'):
    global fft_fwd, fft_rev
    if fft_fwd is None:
        print('Planning FFT... ',end='')
        
        pyfftw.config.NUM_THREADS = 8
        #pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE' #'FFTW_MEASURE'

        try:
            with open(wisdom_file,'r') as f:
                txt = f.read()
            wisdom = tuple(txt.encode().split(b'\n###\n'))
            pyfftw.import_wisdom(wisdom)
        except(IOError, IndexError):
            pass
        N_FFT = 2 * pyfftw.next_fast_len(Nv + 1)
        S_klm_in  = pyfftw.empty_aligned((N_FFT,   NwG, NwL), dtype='float32')
        S_klm_out = pyfftw.empty_aligned((N_FFT//2 + 1, NwG, NwL), dtype='complex64')
        fft_fwd   = pyfftw.FFTW(S_klm_in, S_klm_out, axes=(0,), flags=(patience,))

        ##    ## For now we just use numpy's FFT for the reverse since it's not a bottleneck...
        ##    S_k_in    = pyfftw.empty_aligned( Nv + 1, dtype='complex64')
        ##    S_k_out   = pyfftw.empty_aligned( 2*Nv,   dtype='float32')
        ##    fft_rev   = pyfftw.FFTW(S_k_in, S_k_out, flags=(patience,), direction='FFTW_BACKWARD')
        
        fft_rev = None

        wisdom = pyfftw.export_wisdom()
        with open(wisdom_file,'w') as f:
            f.write(b'\n###\n'.join(wisdom).decode())
        print('Done!')

        
        print('              Matrix (ms):  Transform (ms):')


def synthesize_spectrum(v, v0i, log_wGi, log_wLi, S0i,
                        dxG = 0.1, dxL = 0.1,
                        f_calc_matrix=calc_matrix,
                        f_apply_transform=apply_transform,
                        plan_only=False):
    
    idx = (v0i >= np.min(v)) & (v0i < np.max(v))
    
    v0i = v0i[idx].astype(np.float32)
    log_wGi = log_wGi[idx].astype(np.float32)
    log_wLi = log_wLi[idx].astype(np.float32)
    S0i = S0i[idx].astype(np.float32)

    log_wG = init_w_axis(dxG,log_wGi) #Eq 3.8
    log_wL = init_w_axis(dxL,log_wLi) #Eq 3.9

    plan_FFTW(v.size, log_wG.size, log_wL.size, patience='FFTW_PATIENT')
    if plan_only:
        return
    
    t_list = []
    t_list.append(perf_counter())
    
    S_klm = f_calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i)
    t_list.append(perf_counter())
    
    I = f_apply_transform(v, log_wG, log_wL, S_klm)
    t_list.append(perf_counter())
    
    return I, S_klm, t_list

if __name__ == '__main__':
    import compare_functions
