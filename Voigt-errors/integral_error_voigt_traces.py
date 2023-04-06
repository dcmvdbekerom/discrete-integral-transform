import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import sys
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.optimize import minimize
import pandas as pd

f_GG0 = lambda x: np.exp(-(np.pi*x)**2/(4*np.log(2)))
f_GG1 = lambda x: -(np.pi*x)**2/(2*np.log(2)) * f_GG0(x)
f_GG2 = lambda x: (np.pi*x)**2/(2*np.log(2))*((np.pi*x)**2/(2*np.log(2))-2) * f_GG0(x)

f_GL0 = lambda x: np.exp(-np.abs(np.pi*x))
f_GL1 = lambda x: -np.pi*np.abs(x) * f_GL0(x)
f_GL2 = lambda x: np.pi*np.abs(x)*(np.pi*np.abs(x)-1) * f_GL0(x)

C1 = 1.06920
C2 = 0.86639

f_aG  = lambda d: 2*(1-d)/(C1*(1+d)+(C2*(1+d)**2+4*(1-d)**2)**0.5)
f_aL  = lambda d: 2*(1+d)/(C1*(1+d)+(C2*(1+d)**2+4*(1-d)**2)**0.5)

f_GV0 = lambda x,d: f_GG0(x*f_aG(d)) * f_GL0(x*f_aL(d))

##def gV(v,d):
##    dv = (v[-1]-v[0])/(v.size-1)
##    x = np.arange(v.size + 1) / (2 * v.size * dv)
##    I = np.fft.ifftshift(np.fft.irfft(gV_FT(x,d)))[v.size//2:-v.size//2]
##    return I/np.max(I)


dx = 0.01
da = 0.001
x_max = 10.0
x = np.arange(-x_max,x_max,dx)

pG = 1.1
pL = 1.1

d = 0.0 # G = -1..1 = L

#calc exact function:
Ie = f_GV0(x,d)

#calc prototypes:
aG = f_aG(d)
aL = f_aL(d)

I_G0 = f_GG0(aG*x)
I_G1 = f_GG1(aG*x)
I_G2 = f_GG2(aG*x)

I_L0 = f_GL0(aL*x)
I_L1 = f_GL1(aL*x)
I_L2 = f_GL2(aL*x)

RMS_L1L1 = I_G0**2 *   I_L1**2
RMS_L1L2 = I_G0**2 * 2*I_L1*I_L2
RMS_L2L2 = I_G0**2 *   I_L2**2

RMS_G1L1 = 2*I_G0*I_L0 * I_G1*I_L1
RMS_G1L2 = 2*I_G0*I_L0 * I_G1*I_L2
RMS_G2L1 = 2*I_G0*I_L0 * I_G2*I_L1
RMS_G2L2 = 2*I_G0*I_L0 * I_G2*I_L2

RMS_G1G1 = I_L0**2 * I_G1**2 
RMS_G1G2 = I_L0**2 * 2*I_G1*I_G2
RMS_G2G2 = I_L0**2 * I_G2**2



def errfun(awG,awL,tauG,tauL):
    global Ie,Ia_G0,Ia_G1,Ia_L0,Ia_L1,pG,pL
    #calc approximation:
    Ia   = ((1-awG)*(1-awL)*Ia_G0*Ia_L0 +
            (1-awG)*(  awL)*Ia_G0*Ia_L1 +
            (  awG)*(1-awL)*Ia_G1*Ia_L0 +
            (  awG)*(  awL)*Ia_G1*Ia_L1 )
            
    
    plt.plot(x,(Ia - Ie)**2)
    e_2  = np.sqrt(np.sum((Ie - Ia)**2)*dx)

    #Approximated error:
    
    cG1 =     (tauG - awG)                *np.log(1/pG)
    cG2 = 0.5*(tauG**2 - 2*awG*tauG + awG)*np.log(1/pG)**2
    cL1 =     (tauL - awL)                *np.log(1/pL)
    cL2 = 0.5*(tauL**2 - 2*awL*tauL + awL)*np.log(1/pL)**2

    E2a = ( cL1*cL1 * I_G0**2 *   I_L1**2 +
            cL1*cL2 * I_G0**2 * 2*I_L1*I_L2 +
            cL2*cL2 * I_G0**2 *   I_L2**2 +

            cG1*cL1 * 2*I_G0*I_L0 * I_G1*I_L1 +
            cG1*cL2 * 2*I_G0*I_L0 * I_G1*I_L2 +
            cG2*cL1 * 2*I_G0*I_L0 * I_G2*I_L1 +
            cG2*cL2 * 2*I_G0*I_L0 * I_G2*I_L2 +

            cG1*cG1 * I_L0**2 * I_G1**2 +
            cG1*cG2 * I_L0**2 * 2*I_G1*I_G2 +
            cG2*cG2 * I_L0**2 * I_G2**2 +
            0.0)

    plt.plot(x,E2a,'k--')
    
    
    #calc error:

    return e_2

tauG_arr  = np.linspace(0.0,1.0,7)
tauL_arr  = np.linspace(0.0,1.0,7)
awG_kinds = {'linear':'+','ZEP':'x','min-RMS':'.'}
awL_kinds = {'linear':'s','ZEP':'D','min-RMS':'o'}
df = pd.DataFrame(columns=['awG_kind','awL_kind','awG','awL'])


##tauGi = 0.2
##tauLi = 0.7
for tauGi in tauG_arr[1:-1]:
    for tauLi in tauL_arr[1:-1]:
        print(tauGi,tauLi)
        #calc function bases:
        Ia_G0 = f_GG0(x*aG*pG**  -tauGi )
        Ia_G1 = f_GG0(x*aG*pG**(1-tauGi))
        Ia_L0 = f_GL0(x*aL*pL**  -tauLi )
        Ia_L1 = f_GL0(x*aL*pL**(1-tauLi))

        awG_list = [tauGi,
                    (1-pG**-tauGi)/(1-pG**-1),
                    0.5*(tauGi + (1-pG**-tauGi)/(1-pG**-1))]

        awL_list = [tauLi,
                    (1-pL**-tauLi)/(1-pL**-1),
                    0.5*(tauLi + (1-pL**-tauLi)/(1-pL**-1))]

        for awG_kind, awG in zip(awG_kinds.keys(),awG_list):
            for awL_kind, awL in zip(awL_kinds.keys(),awL_list):
                df.loc[df.shape[0]] = [awG_kind,
                                       awL_kind,
                                       awG,
                                       awL]
                errfun(awG,awL,tauGi,tauLi)
        plt.xlim(-2,2)
        plt.show()
        
##        awG0 = 0.5*(tauGi + (1-pG**-tauGi)/(1-pG**-1))
##        awL0 = 0.5*(tauLi + (1-pL**-tauLi)/(1-pL**-1))
##
##        awG_arr = np.arange(awG0-0.05,awG0+0.05,da)
##        awL_arr = np.arange(awL0-0.05,awL0+0.05,da)
##        X, Y = np.meshgrid(awL_arr,awG_arr)
##        Z= np.zeros((awG_arr.size,awL_arr.size))
##        for i in range(awG_arr.size):
##            for j in range(awL_arr.size):
##                Z[i,j] = errfun(awG_arr[i],awL_arr[j])
##
##        CS = plt.contour(Y,X,Z,levels = 20,colors='k',linestyles='dashed')


size = 7

for key in awG_kinds.keys():
    x = df.awG[df.awG_kind==key]
    y = df.awL[df.awG_kind==key]
    marker = awG_kinds[key]
    plt.plot(x,y,marker+'k',fillstyle='full',markersize=size,label='G-'+key)

for key in awL_kinds.keys():
    x = df.awG[df.awL_kind==key]
    y = df.awL[df.awL_kind==key]
    marker = awL_kinds[key]
    plt.plot(x,y,marker+'k',fillstyle='none',markersize=size,label='L-'+key)

plt.xlabel('$a_wG$')
plt.ylabel('$a_wL$')
plt.legend(loc=2)
##plt.xlim(0,1)
##plt.ylim(0,1)
plt.show()
