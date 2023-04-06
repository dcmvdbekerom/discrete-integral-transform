import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import sys
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.optimize import minimize
import pandas as pd
from scipy.special import erfcx

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

c_arr = np.array(   [
                    [0,0,1/2,-1,1,0,0,0,0],
                    [0,0,1/2,-2,5/2,-1,1,0,0],
                    [0,0,1/2,-3,19/4,-9/2,5,-1,1],
                    [0,1,-3/2,1,-1,0,0,0,0],
                    [0,1,-9/4,7/2,-4,1,-1,0,0],
                    [0,0,3/4,-5/2,3,-1,1,0,0],
                    [0,0,9/8,-23/4,33/4,-6,13/2,-1,1],
                    [3/4,-5/2,3,-1,1,0,0,0,0],
                    [-3/8,13/4,-21/4,5,-11/2,1,-1,0,0],
                    [33/16,-95/8,39/2,-89/4,53/2,-19/2,10,-1,1]
                    ])

da = 0.001

pG = 1.2
pL = 1.2

log_pG = np.log(pG)
log_pL = np.log(pL)

d = 0.0 # G = -1..1 = L

#calc prototypes:
x = (1 + d)/(1 - d) * (2*np.log(2))**0.5
a = np.pi*f_aG(d)/(2*np.log(2))**0.5

F_arr = np.array([(1 if n%2 else np.pi**0.5*erfcx(x))*x**n/a for n in range(9)])
S_arr = c_arr.dot(F_arr)
(S_L1L1,S_L1L2,S_L2L2,S_G1L1,S_G1L2,S_G2L1,S_G2L2,S_G1G1,S_G1G2,S_G2G2) = S_arr

def fitfun(C,tauG,tauL):

    cGG,cGL,cLG,cLL = C

    awG0 = 0.5*(tauG + (1-pG**-tauG)/(1-pG**-1))
    awL0 = 0.5*(tauL + (1-pL**-tauL)/(1-pL**-1))
    awG  = awG0 + cGG*tauG*(1-tauG) + cGL*tauL*(1-tauL)
    awL  = awL0 + cLG*tauG*(1-tauG) + cLL*tauL*(1-tauL)

    E_RMS = np.sum(errfun(awG,awL,tauG,tauL)**2)**0.5
    
    return E_RMS
    

def errfun(awG,awL,tauG,tauL):

    cG1 =     (awG     -       tauG)*np.log(pG)
    cL1 =     (awL     -       tauL)*np.log(pL)
    cG2 = 0.5*(tauG**2 - 2*awG*tauG + awG)*np.log(pG)**2
    cL2 = 0.5*(tauL**2 - 2*awL*tauL + awL)*np.log(pL)**2

    E_RMS = (   cL1*cL1 * S_L1L1 +
              2*cL1*cL2 * S_L1L2 +
                cL2*cL2 * S_L2L2 +
  
              2*cG1*cL1 * S_G1L1 +
              2*cG1*cL2 * S_G1L2 +
              2*cG2*cL1 * S_G2L1 +
              2*cG2*cL2 * S_G2L2 +

                cG1*cG1 * S_G1G1 +
              2*cG1*cG2 * S_G1G2 +
                cG2*cG2 * S_G2G2 )**0.5

    return E_RMS

N_grid = 5

tauG_arr  = np.linspace(0.0,1.0,N_grid)
tauL_arr  = np.linspace(0.0,1.0,N_grid)
awG_kinds = {'linear':'+'}#,'ZEP':'x','min-RMS':'.'}
awL_kinds = {'linear':'s'}#,'ZEP':'D','min-RMS':'o'}
df = pd.DataFrame(columns=['awG_kind','awL_kind','awG','awL'])


X_dict = {}
Y_dict = {}

for tauG in tauG_arr:
    for tauL in tauL_arr:
        awG0 = tauG
        awL0 = tauL

        awG_arr = np.arange(awG0 - 0.1,awG0 + 0.1,da)
        awL_arr = np.arange(awL0 - 0.15,awL0 + 0.05,da)
        X, Y = np.meshgrid(awG_arr,awL_arr)
        X_dict[(tauG,tauL)] = X
        Y_dict[(tauG,tauL)] = Y

for tauG in tauG_arr:
    for tauL in tauL_arr:

        awG = (tauG +
               0.5*(log_pG*tauG*(1 - tauG)*(S_G1L1*S_G2L1 - S_G1G2*S_L1L1) +
                    log_pL**2/log_pG*tauL*(1 - tauL)*(S_G1L1*S_L1L2 - S_G1L2*S_L1L1))
                    /
                   (S_G1G1*S_L1L1 - S_G1L1**2 +
                    log_pL*(1 - 2*tauL)*(S_G1G1*S_L1L2 - S_G1L1*S_G1L2) +
                    log_pG*(1 - 2*tauG)*(S_G1G2*S_L1L1 - S_G1L1*S_G2L1))
               )
        
        awL = (tauL +
               0.5*(log_pL*tauL*(1 - tauL)*(S_G1L1*S_G1L2 - S_G1G1*S_L1L2) +
                    log_pG**2/log_pL*tauG*(1 - tauG)*(S_G1G2*S_G1L1 - S_G1G1*S_G2L1))
                    /
                   (S_G1G1*S_L1L1 - S_G1L1**2 +
                    log_pL*(1 - 2*tauL)*(S_G1G1*S_L1L2 - S_G1L1*S_G1L2) +
                    log_pG*(1 - 2*tauG)*(S_G1G2*S_L1L1 - S_G1L1*S_G2L1))
               )
        
        awG_list = [tauG,awG]
        awL_list = [tauL,awL]

        for awG_kind, awG in zip(awG_kinds.keys(),awG_list):
            for awL_kind, awL in zip(awL_kinds.keys(),awL_list):
                df.loc[df.shape[0]] = [awG_kind,
                                       awL_kind,
                                       awG,
                                       awL]
        
        X = X_dict[(tauG,tauL)]
        Y = Y_dict[(tauG,tauL)]
        Z = errfun(X,Y,tauG,tauL)
        CS = plt.contour(X,Y,Z,levels = 20)

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
