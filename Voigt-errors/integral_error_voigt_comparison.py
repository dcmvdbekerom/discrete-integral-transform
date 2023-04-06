import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import sys
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.optimize import minimize
import pandas as pd
import voigtlib

W = voigtlib.WeightOptimizer()

f_GG0 = lambda x: np.exp(-(np.pi*x)**2/(4*np.log(2)))
f_GG1 = lambda x: -(np.pi*x)**2/(2*np.log(2)) * f_GG0(x)
f_GG2 = lambda x: (np.pi*x)**2/(2*np.log(2))*((np.pi*x)**2/(2*np.log(2))-2) * f_GG0(x)

f_GL0 = lambda x: np.exp(-np.abs(np.pi*x))
f_GL1 = lambda x: -np.pi*x * f_GL0(x)
f_GL2 = lambda x: np.pi*x*(np.pi*x-1) * f_GL0(x)

C1 = 1.06920
C2 = 0.86639

f_aG  = lambda d: 2*(1-d)/(C1*(1+d)+(C2*(1+d)**2+4*(1-d)**2)**0.5)
f_aL  = lambda d: 2*(1+d)/(C1*(1+d)+(C2*(1+d)**2+4*(1-d)**2)**0.5)

f_GV0 = lambda x,d: f_GG0(x*f_aG(d)) * f_GL0(x*f_aL(d))


dx = 0.01
da = 0.001
x_max = 10.0
x = np.arange(0,x_max,dx)

pG = 1.1
pL = 1.1

d = 0.2 # G = -1..1 = L
W.calc_from_d(d)

#calc prototypes:
aG = f_aG(d)
aL = f_aL(d)

I_G0 = f_GG0(aG*x)
I_G1 = f_GG1(aG*x)
I_G2 = f_GG2(aG*x)

I_L0 = f_GL0(aL*x)
I_L1 = f_GL1(aL*x)
I_L2 = f_GL2(aL*x)

SSq_L1L1 = np.sum(I_G0**2 *   I_L1**2)
SSq_L1L2 = np.sum(I_G0**2 * 2*I_L1*I_L2)
SSq_L2L2 = np.sum(I_G0**2 *   I_L2**2)

SSq_G1L1 = np.sum(2*I_G0*I_L0 * I_G1*I_L1)
SSq_G1L2 = np.sum(2*I_G0*I_L0 * I_G1*I_L2)
SSq_G2L1 = np.sum(2*I_G0*I_L0 * I_G2*I_L1)
SSq_G2L2 = np.sum(2*I_G0*I_L0 * I_G2*I_L2)

SSq_G1G1 = np.sum(I_L0**2 * I_G1**2)
SSq_G1G2 = np.sum(I_L0**2 * 2*I_G1*I_G2)
SSq_G2G2 = np.sum(I_L0**2 * I_G2**2)

def errfun(awG,awL,tauG,tauL):

    cG1 =     (                tauG - awG)*np.log(1/pG)
    cL1 =     (                tauL - awL)*np.log(1/pL)
    cG2 = 0.5*(tauG**2 - 2*awG*tauG + awG)*np.log(1/pG)**2
    cL2 = 0.5*(tauL**2 - 2*awL*tauL + awL)*np.log(1/pL)**2

    E_RMS = ( cL1*cL1 * SSq_L1L1 +
              cL1*cL2 * SSq_L1L2 +
              cL2*cL2 * SSq_L2L2 +
  
              cG1*cL1 * SSq_G1L1 +
              cG1*cL2 * SSq_G1L2 +
              cG2*cL1 * SSq_G2L1 +
              cG2*cL2 * SSq_G2L2 +

              cG1*cG1 * SSq_G1G1 +
              cG1*cG2 * SSq_G1G2 +
              cG2*cG2 * SSq_G2G2 )**0.5

    return E_RMS

Ie = f_GV0(x,d)

def errfun_old(awG,awL):
    global Ie,Ia_G0,Ia_G1,Ia_L0,Ia_L1,pG,pL
    #calc approximation:
    Ia   = ((1-awG)*(1-awL)*Ia_G0*Ia_L0 +
            (1-awG)*(  awL)*Ia_G0*Ia_L1 +
            (  awG)*(1-awL)*Ia_G1*Ia_L0 +
            (  awG)*(  awL)*Ia_G1*Ia_L1 )
   
    e_2  = np.sqrt(np.sum((Ie - Ia)**2)*dx)

    return e_2




N_grid = 5

tauG_arr  = np.linspace(0.0,1.0,N_grid)
tauL_arr  = np.linspace(0.0,1.0,N_grid)
awG_kinds = {'linear':'+','ZEP':'x','opti':'.'}
awL_kinds = {'linear':'s','ZEP':'D','opti':'o'}
df = pd.DataFrame(columns=['awG_kind','awL_kind','awG','awL'])


##tauG = 0.2
##tauL = 0.7
for tauG in tauG_arr:
    for tauL in tauL_arr:
        print(tauG,tauL)
        #calc function bases:
        Ia_G0 = f_GG0(x*aG*pG**  -tauG )
        Ia_G1 = f_GG0(x*aG*pG**(1-tauG))
        Ia_L0 = f_GL0(x*aL*pL**  -tauL )
        Ia_L1 = f_GL0(x*aL*pL**(1-tauL))

        awG0,awL0 = W.weights(tauG,tauL,np.log(pG),np.log(pL))

        awG_list = [tauG,
                    (1-pG**-tauG)/(1-pG**-1),
                    awG0]

        awL_list = [tauL,
                    (1-pL**-tauL)/(1-pL**-1),
                    awL0]

        for awG_kind, awG in zip(awG_kinds.keys(),awG_list):
            for awL_kind, awL in zip(awL_kinds.keys(),awL_list):
                df.loc[df.shape[0]] = [awG_kind,
                                       awL_kind,
                                       awG,
                                       awL]
                errfun(awG,awL,tauG,tauL)

        awG_arr = np.arange(awG0 - 0.05,awG0 + 0.05,da)
        awL_arr = np.arange(awL0 - 0.05,awL0 + 0.05,da)
        X, Y = np.meshgrid(awG_arr,awL_arr)


        Z= np.zeros((awL_arr.size,awG_arr.size))
        for i in range(awG_arr.size):
            for j in range(awL_arr.size):
                Z[j,i] = errfun_old(awG_arr[i],awL_arr[j])

        CS = plt.contour(X,Y,Z,levels = 20)#,colors='r',linestyles='dashed')

        
        Z2 = errfun(X,Y,tauG,tauL)
        CS2 = plt.contour(X,Y,Z2,levels = 20,colors='k',linestyles='dashed')


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
