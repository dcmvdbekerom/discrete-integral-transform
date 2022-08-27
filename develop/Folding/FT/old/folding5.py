
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, log, exp, pi
from mpmath import jtheta
import sys

gG = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)

def theta2(z,tau):
    return jtheta(3, z, exp(1j*pi*tau))


##x = np.arange(-10,10,0.01)
##y = np.array([float(theta2(0,xi).real) for xi in x])
##
##plt.plot(x,y)
##plt.show()
##sys.exit()




##def gS(v,v_max,w):
##    # z is already constrained by |Re(z)| <= 1/2 and |Im(z)| <= Im(tau)/2
##    # tau should be constrained by F e |Re(tau)| <= 1/2, Im(tau) > 0 and |tau| > 1
##    tau = 4j*log(2)/pi * (2*v_max/w)**2 #imag (0j..+ooj)
##    z = v/v_max * tau / 2 #imag (0j..tau/2)
##    q = exp(1j*pi*tau) #real (0..1)
##    res = jtheta(3,z*pi,q) * gG(v,0.0,w)
##    return res


def gS(v,v_max,w):
    tau = 4j*log(2)/pi * (2*v_max/w)**2 #imag (0j..+ooj)
    z = v/v_max * tau / 2 #imag (0j..tau/2)
    alpha = 1.0

    alpha = (-1j*tau)**0.5 * exp(1j*pi*z**2/tau)    
    tau = 1j*pi/(4*log(2)) * (w/(2*v_max))**2 # tau_p = -1/tau
    z = v/(2*v_max) #z_p = z/tau,  real (0..1/2)

    q = exp(1j*pi*tau) #real (0..1)
    res = jtheta(3,z*pi,q) * gG(v,0.0,w) / alpha
    return res


dv = 0.001
N = 1000
v_max = N*dv
vh = v_max/2
Nk = 20
##w = 1.0

v_arr = np.arange(-v_max,v_max,dv)


plt.axhline(0,c='k')
plt.axvline(0,c='k')

for w in [0.1,0.2,0.5,1.0,2.0,5.0]:
    print('w:',w)
    I = gG(v_arr,0.0,w)
    S = np.sum([gG(v_arr,-2*k*v_max,w) + gG(v_arr,2*k*v_max,w) for k in range(1,Nk)],0)
    S0 = float(gS(0.0,v_max,w).real)
    Sh = float(gS(vh,v_max,w).real)
    S1 = float(gS(v_max,v_max,w).real)

    c = plt.plot(v_arr, I, label=w)[0].get_c()
    plt.plot(v_arr, S+I, '--', c=c)
    plt.plot([-v_max,-vh,0.0,vh,v_max],[S1,Sh,S0,Sh,S1],'o',c=c, mfc=c, mec='k')
    print('')
##plt.yscale('log')
plt.legend()
plt.show()


