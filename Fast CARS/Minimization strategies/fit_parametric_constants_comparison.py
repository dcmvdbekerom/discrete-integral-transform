import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import erfcx
import sys

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

    cG1 = (awG - tauG)*DxG
    cL1 = (awL - tauL)*DxL
    cG2 = 0.5*(tauG**2 - 2*awG*tauG + awG)*DxG**2
    cL2 = 0.5*(tauL**2 - 2*awL*tauL + awL)*DxL**2

    E_RMS2 = ( cL1*cL1 *   S_L1L1 +
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

DxG = 0.2
DxL = 0.2

pG = np.exp(DxG)
pL = np.exp(DxL)

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

labels = [   '$R_{L1L1}$',
             '$R_{L1L2}$',
             '$R_{L2L2}$',
             
             '$R_{G1L1}$',
             '$R_{G1L2}$',
             '$R_{G2L1}$',
             '$R_{G2L2}$',

             '$R_{G1G1}$',
             '$R_{G1G2}$',
             '$R_{G2G2}$',
            ]

##plt.axvline(-1,c='gray',ls='--')
##plt.axvline(1,c='gray',ls='--')
##plt.axhline(0,c='k')
##for i in range(10):
##    plt.plot(d_arr,S_arr[i],label=labels[i])
##    
##
####x_arr = (1+d_arr)/(1-d_arr) * (2*np.log(2))**0.5
####a_arr = np.pi*f_aG(d_arr)/(2*np.log(2))**0.5
####
####F_arr = np.array([(1 if n%2 else np.pi**0.5*erfcx(x_arr))*x_arr**n/a_arr for n in range(9)])
####SSq_arr = c_arr.dot(F_arr)
####
####for i in range(SSq_arr.shape[0]):
####    plt.plot(d_arr,SSq_arr[i],'k--',alpha=0.25)
##
##plt.grid(True)   
##plt.ylim(-0.1,0.2)
##plt.legend()
##plt.xlabel('$Gaussian\\quad\\quad \\leftarrow \\quad\\quad\\quad d \\quad\\quad\\quad \\rightarrow \\quad\\quad Lorentzian$')
##plt.show()

R_G  =  -S_G1G2 / S_G1G1
R_L  =  -S_L1L2 / S_L1L1

S_V  = (S_G1G1*S_L1L1 - S_G1L1**2)
R_GG = (S_G1L1*S_G2L1 - S_G1G2*S_L1L1)/S_V
R_GL = (S_G1L1*S_L1L2 - S_G1L2*S_L1L1)/S_V
R_LL = (S_G1L1*S_G1L2 - S_G1G1*S_L1L2)/S_V
R_LG = (S_G1G2*S_G1L1 - S_G1G1*S_G2L1)/S_V

x_arr = (1+d_arr)/(1-d_arr) * (2*np.log(2))**0.5

##plt.axvline(-1,c='gray',ls='--')
##plt.axvline(1,c='gray',ls='--')
##plt.axhline(0,c='k')
##plt.plot(d_arr,R_G,'--',label='$R_G$')
##plt.plot(d_arr,R_L,'--',label='$R_L$')
##plt.plot(d_arr,R_GG,label = '$R_{GG}$') 
##plt.plot(d_arr,R_GL,label = '$R_{GL}$')
##plt.plot(d_arr,R_LL,label = '$R_{LL}$')
##plt.plot(d_arr,R_LG,label = '$R_{LG}$')
##
##plt.xlabel('$Gaussian\\quad\\quad \\leftarrow \\quad\\quad\\quad d \\quad\\quad\\quad \\rightarrow \\quad\\quad Lorentzian$')
##
##plt.grid()
##plt.ylim(-4,4)
##plt.legend()
##plt.show()

for DxG in [0.2]:#[0.2,0.1,0.05,0.02]:
    C_list = []
    err_list = []

##    DxL = 3*DxG

    for d in d_arr:
        
        #calc prototypes:
        aG = f_aG(d)
        aL = f_aL(d)

        I_G0 = f_GG0(aG*x)
        I_G1 = f_GG1(aG*x)
        I_G2 = f_GG2(aG*x)

        I_L0 = f_GL0(aL*x)
        I_L1 = f_GL1(aL*x)
        I_L2 = f_GL2(aL*x)

        S_L1L1 = np.sum(I_G0**2 * I_L1**2)*dx
        S_L1L2 = np.sum(I_G0**2 * I_L1*I_L2)*dx
        S_L2L2 = np.sum(I_G0**2 * I_L2**2)*dx

        S_G1L1 = np.sum(I_G0*I_L0 * I_G1*I_L1)*dx
        S_G1L2 = np.sum(I_G0*I_L0 * I_G1*I_L2)*dx
        S_G2L1 = np.sum(I_G0*I_L0 * I_G2*I_L1)*dx
        S_G2L2 = np.sum(I_G0*I_L0 * I_G2*I_L2)*dx

        S_G1G1 = np.sum(I_L0**2 * I_G1**2)*dx
        S_G1G2 = np.sum(I_L0**2 * I_G1*I_G2)*dx
        S_G2G2 = np.sum(I_L0**2 * I_G2**2)*dx

        R_G  =  -S_G1G2 / S_G1G1
        R_L  =  -S_L1L2 / S_L1L1

        S_V  = (S_G1G1*S_L1L1 - S_G1L1**2)
        R_GG = (S_G1L1*S_G2L1 - S_G1G2*S_L1L1)/S_V
        R_GL = (S_G1L1*S_L1L2 - S_G1L2*S_L1L1)/S_V
        R_LL = (S_G1L1*S_G1L2 - S_G1G1*S_L1L2)/S_V
        R_LG = (S_G1G2*S_G1L1 - S_G1G1*S_G2L1)/S_V

        ## X and Y take the form of t

        # Simple weights
        aG1 = X
        aL1 = Y

        aG2 = X + 0.25*R_G*X*(1-X)*DxG
        aL2 = Y + 0.25*R_L*Y*(1-Y)*DxL

        aG3 = X + 0.5*R_G*X*(1-X)*DxG
        aL3 = Y + 0.5*R_L*Y*(1-Y)*DxL

        aG4 = X + (R_GG*X*(1-X)*DxG**2 + R_GL*Y*(1-Y)*DxL**2)/(2*DxG)
        aL4 = Y + (R_LL*Y*(1-Y)*DxL**2 + R_LG*X*(1-X)*DxG**2)/(2*DxL)

        err_list.append([errfun(aG1,aL1,X,Y),
                         errfun(aG2,aL2,X,Y),
                         errfun(aG3,aL3,X,Y),
                         errfun(aG4,aL4,X,Y),])

    err_arr = np.array(err_list).T

    colors = ['tab:blue','tab:orange','tab:green','tab:red']
    labels = ['$linear$','$G/L-edge\\/only$','$G/L-optimized$','$V-optimized$']
##    styles = {0.2:'-',0.1:'--',0.05:'.-'}
    plt.axhline(0,c='k')
    plt.axvline(-1,c='gray',ls='--')
    plt.axvline(1,c='gray',ls='--')

    for i in range(err_arr.shape[0]):
        plt.plot(d_arr,err_arr[i],c=colors[i],label=labels[i])
##    plt.yscale('log')
    plt.grid()
    plt.ylabel('RMS-error')
    plt.xlabel('$Gaussian\\quad\\quad \\leftarrow \\quad\\quad\\quad d \\quad\\quad\\quad \\rightarrow \\quad\\quad Lorentzian$')
    plt.legend()
plt.show()


