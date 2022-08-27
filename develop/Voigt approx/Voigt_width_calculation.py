
import numpy as np
from numpy import sqrt, log, pi, exp, sin
import matplotlib.pyplot as plt
from scipy.special import wofz
from scipy.optimize import minimize

def Voigt(x, wG, wL):
    gamma = wL/2
    sigma = wG/(2*sqrt(2*log(2)))
    z = (x + 1j*gamma)/(sigma*sqrt(2))
    V = wofz(z).real/(sigma*sqrt(2*pi))
    return V

def Voigt_d(x, d):
    gamma = (1 + d)/4
    sigma = (1 - d)/(4*sqrt(2*log(2)))
    z = (x + 1j*gamma)/(sigma*sqrt(2))
    V = wofz(z).real/(sigma*sqrt(2*pi))
    return V

def get_R(d,R0=None):
    if R0 is None:
        R0 = sqrt((1 + d**2) / 2)
        
    def FWHM_finder(x,d):
        return (Voigt_d(x,d) / Voigt_d(0,d) - 0.5)**2
    
    R = 2*np.abs(minimize(FWHM_finder, R0, d, method='Powell').x[0])
    
    return R

def approx_R(d):
    alpha = 0.18121
    beta = 0.023665*exp(0.6*d) + 0.00418*exp(-1.9*d)
    R = 1 - alpha*(1 - d**2) - beta*sin(pi*d)
    return R


dd = 0.01
d_arr = np.arange(dd - 1, 1 - dd, dd)
R0_arr = sqrt((1 + d_arr**2) / 2)

R_arr = np.array([get_R(d) for d in d_arr])
err = approx_R(d_arr) - R_arr

plt.axvline(-1,c='k',ls='--')
plt.axvline(1,c='k',ls='--')

##plt.axhline(0,c='k')
##plt.plot(d_arr, err)

plt.plot(d_arr, R_arr)
plt.plot(d_arr, approx_R(d_arr), 'k--')

plt.show()
    
    
