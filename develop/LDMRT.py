import numpy as np
from HITEMP_spectra import h,c,k,c_cm,c2,init_database
import HITEMP_spectra as HT
import sys
import matplotlib.pyplot as plt

def calc_LDM(data,v_ax,da_ax,log2vMm_ax,log2gs_ax,na_ax,E_ax):
  
    v_dat      = v0 #+ p*da
    log_wG_dat = log_2vMm #+ hlog_T #minmax can be determined at init
    log_wL_dat = log_2gs #+ log_p + na*log_rT #minmax function can be determined at init
    I_dat      = S0/gr  #* N * (np.exp(-c2T*El) - np.exp(-c2T*Eu)) / np.log(10)  / (gr*Qr(T)*Qv(T,T))

    #Start with [v0,da,log2vMm,na,E]-matrix:
    S_klm = np.zeros((2 * v_ax.size,da_ax.size,log2vMm_ax.size,log2gs_ax,na_ax,E_ax))
    

def calc_stick_spectrum(p,T):
        

    #Each iteration, but only on scalar:
    c2T       = h*c_cm/(k*T)  #scalar
    log_p     = np.log(p)      #scalar
    log_rT    = np.log(296./T) #scalar
    hlog_T    = 0.5*np.log(T)  #scalar
    N         = p*1e5 / (1e6 * k * T) #scalar

    



    Dv_dat = p*da
    Dlog_wG_dat = hlog_T
    Dlog_wL_dat = log_p + na*log_rT
    DI_dat = N * (np.exp(-c2T*Elu))  / (Qr(T)*Qv(T,T)) / np.log(10)


    return (v_dat,log_wG_dat,log_wL_dat,I_dat)































## Define constants for optimized weights:
C1_GG = ((6 * np.pi - 16) / (15 * np.pi - 32)) ** (1 / 1.50)
C1_LG = ((6 * np.pi - 16) / 3 * (np.log(2) / (2 * np.pi)) ** 0.5) ** (1 / 2.25)
C2_GG = (2 * np.log(2) / 15) ** (1 / 1.50)
C2_LG = ((2 * np.log(2)) ** 2 / 15) ** (1 / 2.25)


## Prepare width-axes based on number of gridpoints and linewidth data:
def init_axis(dx, x):
    x_min = np.min(x)
    x_max = np.max(x) + 1e-4
    N = np.ceil((x_max - x_min)/dx) + 1
    print(N)
    x_arr = x_min + dx * np.arange(N)
    return x_arr


## Calculate grid indices and grid alignments:   
def get_indices(arr_i, axis):
    pos    = np.interp(arr_i, axis, np.arange(axis.size))
    index  = pos.astype(int)
    t = pos - index
    return (index, index + 1), (t, 1 - t) 


## Calculate lineshape distribution matrix:
def calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i, optimized):
    
    #  Initialize matrix:
    S_klm = np.zeros((2 * v.size, log_wG.size, log_wL.size))

    #  Calculate grid indices and alignment:    
    ki0, ki1, tvi = get_indices(v0i, v)
    li0, li1, tGi = get_indices(log_wGi, log_wG)
    mi0, mi1, tLi = get_indices(log_wLi, log_wL)

    # Calculate weights:
    if optimized:

        dv = (v[-1] - v[0])/(v.size - 1)
        dxvGi = dv / np.exp(log_wGi)
        dxG = (log_wG[-1] - log_wG[0])/(log_wG.size - 1)
        dxL = (log_wL[-1] - log_wL[0])/(log_wL.size - 1)

        alpha_i = np.exp(log_wLi - log_wGi)

        R_Gv = 8 * np.log(2)
        R_GG = 2 - 1 / (C1_GG + C2_GG * alpha_i ** (2 / 1.50)) ** 1.50
        R_GL = -2 * np.log(2) * alpha_i ** 2

        R_LL = 1
        R_LG = 1 / (C1_LG * alpha_i ** (1 / 2.25) + C2_LG * alpha_i ** (4 / 2.25)) ** 2.25
        
        avi = tvi

        aGi = tGi + (R_Gv * tvi * (tvi - 1) * dxvGi**2 +
                     R_GG * tGi * (tGi - 1) * dxG**2 +
                     R_GL * tLi * (tLi - 1) * dxL**2 ) / (2 * dxG)
        
        aLi = tLi + (R_LG * tGi * (tGi - 1) * dxG**2 +
                     R_LL * tLi * (tLi - 1) * dxL**2 ) / (2 * dxL)

    else:
        avi = tvi
        aGi = tGi
        aLi = tLi
    
    # Add lines to matrix:
    np.add.at(S_klm, (ki0, li0, mi0), S0i * (1-avi) * (1-aGi) * (1-aLi))
    np.add.at(S_klm, (ki0, li0, mi1), S0i * (1-avi) * (1-aGi) *    aLi )
    np.add.at(S_klm, (ki0, li1, mi0), S0i * (1-avi) *    aGi  * (1-aLi))
    np.add.at(S_klm, (ki0, li1, mi1), S0i * (1-avi) *    aGi  *    aLi )
    np.add.at(S_klm, (ki1, li0, mi0), S0i *    avi  * (1-aGi) * (1-aLi))
    np.add.at(S_klm, (ki1, li0, mi1), S0i *    avi  * (1-aGi) *    aLi )
    np.add.at(S_klm, (ki1, li1, mi0), S0i *    avi  *    aGi  * (1-aLi))
    np.add.at(S_klm, (ki1, li1, mi1), S0i *    avi  *    aGi  *    aLi )
    
    return S_klm


## Apply transform:
def apply_transform(v,log_wG,log_wL,S_klm):

    dv    = (v[-1] - v[0]) / (v.size - 1)
    x     = np.arange(v.size + 1) / (2 * v.size * dv)

    # Sum over lineshape distribution matrix:
    S_k_FT = np.zeros(v.size + 1, dtype = np.complex64)
    for l in range(log_wG.size):
        for m in range(log_wL.size):
            wG_l,wL_m = np.exp(log_wG[l]),np.exp(log_wL[m])
            gG_FT = np.exp(-(np.pi*x*wG_l)**2/(4*np.log(2)))
            gL_FT = np.exp(- np.pi*x*wL_m)
            gV_FT = gG_FT * gL_FT
            S_k_FT += np.fft.rfft(S_klm[:,l,m]) * gV_FT
    
    return np.fft.irfft(S_k_FT)[:v.size] / dv


## Synthesize spectrum:
def synthesize_spectrum(v, v0i, log_wGi, log_wLi, S0i, dxG = 0.14, dxL = 0.2, optimized = False):

    # Only process lines within range:
    idx = (v0i >= np.min(v)) & (v0i < np.max(v))
    v0i, log_wGi, log_wLi, S0i = v0i[idx], log_wGi[idx], log_wLi[idx], S0i[idx]

    # Initialize width-axes:
    log_wG = init_w_axis(dxG,log_wGi)
    log_wL = init_w_axis(dxL,log_wLi)

    # Calculate matrix & apply transform:
    S_klm = calc_matrix(v, log_wG, log_wL, v0i, log_wGi, log_wLi, S0i, optimized)
    I = apply_transform(v, log_wG, log_wL, S_klm)
        
    return I,S_klm


dv = 0.002
dxG = 0.2
dxL = 0.2
dxE = 0.2


HITEMP_path = "C:/HITEMP/"
init_database([HITEMP_path + "02_2000-2125_HITEMP2010.par",
               HITEMP_path + "02_2125-2250_HITEMP2010.par",
               HITEMP_path + "02_2250-2500_HITEMP2010.par"])

v_min = 2000.0 #cm-1
v_max = 2400.0 #cm-1
dv =     0.001 #cm-1
v = np.arange(v_min,v_max,dv) #cm-1

T0 = 296. #K
E0 = k*T0/(h*c_cm) #cm-1
T_max = 3000#K
p_max = 1.0#bar

v0,da,S0,El,Eu,log2gs,na,log2vMm,gr = HT.data


path = 'C:/CDSD4000/npy/'
def load(path):
    print(path)
    return np.load(path)

v0 = load(path+'v0.npy')
da = load(path+'da.npy')
log2vMm = load(path+'log_2vMm.npy')
log2gs = load(path+'log_2gs.npy')
na = load(path+'na.npy')
El = load(path+'El.npy')
Eu = load(path+'Eu.npy')


N = 10000
plt.plot(da[:N],na[:N],'.')
##plt.plot(v0[:N],da[:N])
plt.show()

v_ax = v
##v0 = v0[(v0 >= v_min)&(v0 < v_max)]
da_ax = init_axis(dv/p_max,da)
log2vMm_ax = init_axis(dxG,log2vMm)
log2gs_ax = init_axis(dxL,log2gs)
na_ax = init_axis(dxL/np.abs(np.log(T0/T_max)),na)
print('---')

Ein = [np.min(El),np.min(Eu),np.max(El),np.max(Eu)]
print(Ein)

E_ax_L = init_axis(dxE*E0,[0,E0])
print(E_ax_L.size)

E_ax_H = np.exp(init_axis(dxE,np.log([E0,np.max(El),np.max(Eu)])))
print(E_ax_H.size)

E_ax = np.array(list(E_ax_L)[:-2] + list(E_ax_H))
print(E_ax.size)


shape = [v_ax.size,da_ax.size,log2vMm_ax.size,log2gs_ax.size,na_ax.size,E_ax.size]
print(shape)
sys.exit()

ki,  aki = get_indices(v0, v_ax)
li,  ali = get_indices(da, da_ax)
mi,  ami = get_indices(log2vMm, log2vMm_ax)
ni,  ani = get_indices(log2gs,  log2gs_ax)
oi,  aoi = get_indices(na, na_ax)
pil, apil = get_indices(El, E_ax)
piu, apiu = get_indices(Eu, E_ax)

# First do a dry run to find indices of non-zero elements:
S_nzi = np.zeros([2**5]+shape[1:],dtype=np.bool)

for l in [0,1]:
    for m in [0,1]:
        for n in [0,1]:
            for o in [0,1]:
                for p in [0,1]:
                    idx = (l<<4) | (m<<3) | (n<<2) | (o<<1) | (p<<0)
                    S_nzi[idx, li[l], mi[m], ni[n], oi[o], pil[p]] = True
                    S_nzi[idx, li[l], mi[m], ni[n], oi[o], piu[p]] = True
                    print(l,m,n,o,p)

# Get list of all nonzero indices:
S_nz = np.zeros(shape[1:],dtype = np.bool)
for i in range(2**5):
    S_nz = S_nz|S_nzi[i]
    
i_nz = np.unravel_index(np.arange(S_nz.size)[S_nz.flatten()],shape[1:])

# Now fill the array:
S_q = np.zeros((2*v_ax.size,i_nz[0].size))

##for l,m,n,o,p in zip(i_nz):
    

# Do the FFT and collapse the log2gs axis:

# That's it for now!
