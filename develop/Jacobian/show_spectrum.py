import sys
sys.path.append('../../demo/')
from HITEMP_spectra import init_database, calc_stick_spectrum
from discrete_integral_transform import synthesize_spectrum, gV
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

def calc_LL(Iexp,I,dv):
    return np.log(np.sum((Iexp - I)**2)*dv)

## Download database files from https://hitran.org/hitemp/

HITEMP_path = "C:/HITEMP/"
init_database([HITEMP_path + "02_2000-2125_HITEMP2010.par",
               HITEMP_path + "02_2125-2250_HITEMP2010.par",
               HITEMP_path + "02_2250-2500_HITEMP2010.par"])

v_min = 2200.0 #cm-1
v_max = 2400.0 #cm-1
dv =     0.002 #cm-1
v = np.arange(v_min,v_max,dv) #cm-1

dxG = 0.2
dxL = 0.2

T = 1000.0 #K
p =    0.1 #bar

data = calc_stick_spectrum(p,T)
idx = np.argsort(data[3])[::-1]
data = [data[0][idx],
        data[1][idx],
        data[2][idx],
        data[3][idx]]


i = 3
j = 0
print(data[i][j])


##v0i,log_wGi,log_wLi,S0i = v0i,log_wGi,log_wLi,S0i
print('{:.2f}M lines '.format(len(data[0])*1e-6))
I0,S_klm,J = synthesize_spectrum(v,*data,dxG=dxG,dxL=dxL,optimized=False)
Iexp = I0 + np.random.normal(0,0.1,I0.size)


I1,S_klm,J1 = synthesize_spectrum(v,*data,Iexp=Iexp,dxG=dxG,dxL=dxL,optimized=False)
y1 = calc_LL(Iexp,I1,dv)

S1 = data[i][j]
S2 = data[i][j]*1.1
data[i][j] = S2

DS = S2-S1
Dy = y2-y1

I2,S_klm, J2 = synthesize_spectrum(v,*data,dxG=dxG,dxL=dxL,optimized=False)
y2 = calc_LL(Iexp,I2,dv)

J_ij = Dy/DS

dIdx = DS * gV(v,data[0][j],np.exp(data[1][j]),np.exp(data[2][j]))
DI = Iexp - I1
Int1 = np.sum(DI*Iexp)*dv
Int2 = np.sum(DI*dIdx)*dv





plt.plot(v,Iexp,'.')
plt.plot(v,I2,'-')
plt.plot(v,I1,'-')

plt.xlim(v_max,v_min)
plt.grid(True,alpha=0.2)
plt.axhline(0,c='k',lw=1)
plt.show()
