import numpy as np
from HITEMP_spectra import h,c,k,c_cm,c2,init_database
import HITEMP_spectra as HT
import sys
import matplotlib.pyplot as plt


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

##v0 = load(path+'v0.npy')
da = load(path+'da.npy')
log2vMm = load(path+'log_2vMm.npy')
log2gs = load(path+'log_2gs.npy')
na = load(path+'na.npy')
##El = load(path+'El.npy')
##Eu = load(path+'Eu.npy')


p =    1e-3 #bar
T = 3000.0 #K

##c2T       = -h*c_cm/(k*T)  #scalar
log_p     = np.log(p)      #scalar
log_rT    = np.log(296./T) #scalar
hlog_T    = 0.5*np.log(T)  #scalar
##N         = p*1e5 / (1e6 * k * T) #scalar


log_wG = log2vMm + hlog_T #minmax can be determined at init
log_wL = log2gs + log_p + na*log_rT #minmax function can be determined at init

delta = da/np.exp(log2gs)

wG = np.exp(log_wG)
wL = np.exp(log_wL)

C1 = 1.06920
C2 = 0.86639
wV = 0.5*(C1*wL + (C2*wL**2 + 4*wG**2)**0.5)

delta_v0 = p*da    
print(np.max(np.abs(delta_v0)/wL))
