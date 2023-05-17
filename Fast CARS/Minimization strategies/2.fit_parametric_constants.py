import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import erfcx
import sys

#these are the 'o' versions (with the scalar factor included).

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

def errfun(awG,awL,tauG,tauL):

    cG1 = (awG - tauG)*np.log(pG)
    cL1 = (awL - tauL)*np.log(pL)
    cG2 = 0.5*(tauG**2 - 2*awG*tauG + awG)*np.log(pG)**2
    cL2 = 0.5*(tauL**2 - 2*awL*tauL + awL)*np.log(pL)**2

    E_RMS2 = (cL1*cL1 *   S_L1L1 +
              cL1*cL2 * 2*S_L1L2 +
              cL2*cL2 *   S_L2L2 +
  
              cG1*cL1 * 2*S_G1L1 +
              cG1*cL2 * 2*S_G1L2 +
              cG2*cL1 * 2*S_G2L1 +
              cG2*cL2 * 2*S_G2L2 +

              cG1*cG1 *   S_G1G1 +
              cG1*cG2 * 2*S_G1G2 +
              cG2*cG2 *   S_G2G2 )**0.5

    return (np.sum(E_RMS2)/E_RMS2.size)**0.5


def fitfun(C,tauG,tauL):

    cGG,cGL,cLG,cLL = C

    awG0 = tauG
    awL0 = tauL

    awG  = awG0 + cGG*tauG*(1-tauG) + cGL*tauL*(1-tauL)
    awL  = awL0 + cLG*tauG*(1-tauG) + cLL*tauL*(1-tauL)

    E_RMS = errfun(awG,awL,tauG,tauL)
    E_RMS = (np.sum(E_RMS**2)/E_RMS.size)**0.5
    
    return E_RMS




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

dx = 0.01
x_max = 10.0
x = np.arange(0,x_max,dx)

N_grid = 10
tau_arr = np.linspace(0.0,1.0,N_grid)
X, Y = np.meshgrid(tau_arr,tau_arr)
d_arr  = np.linspace(-1.0,1.0,101)

pG = 1.2
pL = 1.2

S_list = []

for d in d_arr:
    aG = f_aG(d)
    aL = f_aL(d)

    I_GV = f_GV0(x,d)

    I_G0 = f_GG0(aG*x)
    I_G1 = f_GG1(aG*x)
    I_G2 = f_GG2(aG*x)

    I_L0 = f_GL0(aL*x)
    I_L1 = f_GL1(aL*x)
    I_L2 = f_GL2(aL*x)

    # 2x because x is only positive
    S_L1L1 = 2*np.sum(I_G0**2 * I_L1**2)*dx
    S_L1L2 = 2*np.sum(I_G0**2 * I_L1*I_L2)*dx   
    S_L2L2 = 2*np.sum(I_G0**2 * I_L2**2)*dx

    S_G1L1 = 2*np.sum(I_G0*I_L0 * I_G1*I_L1)*dx
    S_G1L2 = 2*np.sum(I_G0*I_L0 * I_G1*I_L2)*dx
    S_G2L1 = 2*np.sum(I_G0*I_L0 * I_G2*I_L1)*dx
    S_G2L2 = 2*np.sum(I_G0*I_L0 * I_G2*I_L2)*dx

    S_G1G1 = 2*np.sum(I_L0**2 * I_G1**2)*dx
    S_G1G2 = 2*np.sum(I_L0**2 * I_G1*I_G2)*dx
    S_G2G2 = 2*np.sum(I_L0**2 * I_G2**2)*dx

    S_list.append([  S_L1L1,
                     S_L1L2,
                     S_L2L2,
                     
                     S_G1L1,
                     S_G1L2,
                     S_G2L1,
                     S_G2L2,

                     S_G1G1,
                     S_G1G2,
                     S_G2G2,
                    ])


S_arr = np.array(S_list).T

[S_L1L1,
 S_L1L2,
 S_L2L2,
 
 S_G1L1,
 S_G1L2,
 S_G2L1,
 S_G2L2,

 S_G1G1,
 S_G1G2,
 S_G2G2,
] = S_arr

labels = [   'L1L1',
             'L1L2',
             'L2L2',
             
             'G1L1',
             'G1L2',
             'G2L1',
             'G2L2',

             'G1G1',
             'G1G2',
             'G2G2',
            ]
plt.axhline(0,c='k')
for i in range(10):
    plt.plot(d_arr,S_arr[i],label=labels[i])
    

##x_arr = (1+d_arr)/(1-d_arr) * (2*np.log(2))**0.5
##a_arr = np.pi*f_aG(d_arr)/(2*np.log(2))**0.5
##
##F_arr = np.array([(1 if n%2 else np.pi**0.5*erfcx(x_arr))*x_arr**n/a_arr for n in range(9)])
##SSq_arr = c_arr.dot(F_arr)
##
##for i in range(SSq_arr.shape[0]):
##    plt.plot(d_arr,SSq_arr[i],'k--',alpha=0.25)
   
plt.ylim(-0.1,0.2)
plt.legend()
plt.xlabel('$Gaussian\\quad\\quad \\leftarrow \\quad\\quad\\quad d \\quad\\quad\\quad \\rightarrow \\quad\\quad Lorentzian$')
plt.show()

## Error-free up to this point
##---------------------------------------------------------------------------


##
##
##S_V  = (S_G1G1*S_L1L1 - S_G1L1**2)
##S_GG = (S_G1L1*S_G2L1 - S_G1G2*S_L1L1)/(S_G1G1*S_L1L1 - S_G1L1**2)
##S_GL = (S_G1L1*S_L1L2 - S_G1L2*S_L1L1)/(S_G1G1*S_L1L1 - S_G1L1**2)
##S_LL = (S_G1L1*S_G1L2 - S_G1G1*S_L1L2)/(S_G1G1*S_L1L1 - S_G1L1**2)
##S_LG = (S_G1G2*S_G1L1 - S_G1G1*S_G2L1)/(S_G1G1*S_L1L1 - S_G1L1**2)
##x_arr = (1+d_arr)/(1-d_arr) * (2*np.log(2))**0.5
##
##plt.axhline(0,c='k')
##plt.plot(np.log(x_arr),S_GG,label = 'GG') 
##plt.plot(np.log(x_arr),S_GL,label = 'GL')
##plt.plot(np.log(x_arr),S_LL,label = 'LL')
##plt.plot(np.log(x_arr),S_LG,label = 'LG')
##plt.legend()
##plt.show()
##
##
##         
##
##
##
##for pL in [1.05,1.1,1.2,1.5]:
##    C_list = []
##    err_list = []
##    for d in d_arr:
##        
##        #calc prototypes:
##        aG = f_aG(d)
##        aL = f_aL(d)
##
##        I_G0 = f_GG0(aG*x)
##        I_G1 = f_GG1(aG*x)
##        I_G2 = f_GG2(aG*x)
##
##        I_L0 = f_GL0(aL*x)
##        I_L1 = f_GL1(aL*x)
##        I_L2 = f_GL2(aL*x)
##
##        S_L1L1 = np.sum(I_G0**2 * I_L1**2)*dx
##        S_L1L2 = np.sum(I_G0**2 * I_L1*I_L2)*dx
##        S_L2L2 = np.sum(I_G0**2 * I_L2**2)*dx
##
##        S_G1L1 = np.sum(I_G0*I_L0 * I_G1*I_L1)*dx
##        S_G1L2 = np.sum(I_G0*I_L0 * I_G1*I_L2)*dx
##        S_G2L1 = np.sum(I_G0*I_L0 * I_G2*I_L1)*dx
##        S_G2L2 = np.sum(I_G0*I_L0 * I_G2*I_L2)*dx
##
##        S_G1G1 = np.sum(I_L0**2 * I_G1**2)*dx
##        S_G1G2 = np.sum(I_L0**2 * I_G1*I_G2)*dx
##        S_G2G2 = np.sum(I_L0**2 * I_G2**2)*dx
##
####        err = fitfun(C,X,Y)
####        err_list.append(err)
####        print('{:.1f}\t{:.3e}'.format(d,err))
##
##
##    labels = ['$G_G$','$G_L$','$L_G$','$L_L$']
##    colors = ['tab:blue','tab:orange','tab:green','tab:red']
##    plt.axhline(0,c='k')
##    for i in [1]:#range(C_arr.shape[0]):
##        plt.plot(d_arr,C_arr[i],c=colors[i],label=labels[i])
##    plt.plot(d_arr,20*pG**2*(d_arr+1)*100**d_arr*np.log(pL)**2*np.log(pG),'k--')
##    #plt.yscale('log')
##    plt.grid()
##    plt.ylim(-1,1)
##    plt.legend()
##plt.show()
##
####    plt.plot(d_arr,err_list/np.log(pG)**2)
####plt.show()
##
