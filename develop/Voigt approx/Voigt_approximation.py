
import numpy as np
from numpy import sqrt, log, pi, exp, sin
import matplotlib.pyplot as plt
from scipy.special import wofz
from scipy.optimize import minimize, curve_fit

def G(x, wG):
    sigma = wG / (2*sqrt(2*log(2)))
    return 1/(sigma*sqrt(2*pi)) * exp(-x**2/(2*sigma**2))

def L(x, wL):
    gamma = wL / 2
    return gamma / (pi*(x**2 + gamma**2))

def V(x, wG, wL):
    gamma = wL / 2
    sigma = wG / (2*sqrt(2*log(2)))
    z = (x + 1j*gamma) / (sigma*sqrt(2))
    return wofz(z).real / (sigma*sqrt(2*pi))

def V_d(x, d):
    gamma = (1 + d) / 4
    sigma = (1 - d) / (4 * sqrt(2*log(2)))
    z = (x + 1j*gamma) / (sigma * sqrt(2))
    return wofz(z).real / (sigma * sqrt(2*pi))

def V_1(x, d):
    R = get_R(d)
    wG = (1 - d)/(2*R)
    wL = (1 + d)/(2*R)
    return V(x, wG, wL)
    
def get_R(d, R0=None):
    if R0 is None:
        R0 = sqrt((1 + d**2) / 2)
        
    def FWHM_finder(x,d):
        return (V_d(x, d) / V_d(0, d) - 0.5)**2
    
    R = 2*np.abs(minimize(FWHM_finder, R0, d, method='Powell').x[0])
    
    return R

def approx_R(d):
    alpha = 0.18121
    beta = 0.023665*exp(0.6*d) + 0.00418*exp(-1.9*d)
    R = 1 - alpha*(1 - d**2) - beta*sin(pi*d)
    return R

def approx_V(v, wG, wL):
    d = (wL - wG) / (wL + wG)
    wV = approx_R(d) * (wL + wG)
    x = v / wV
    wL_V = wL / wV

    Iw0 = 1 / (wV * (1.065 + 0.447 * wL_V + 0.058 * wL_V**2))
    Iw1 = ((1 - wL_V) * np.exp(-2.772 * x**2) + wL_V * 1 / (1 + 4 * x**2)) * Iw0
    Iw2 = Iw1 + Iw0*0.016*(1-wL_V)*wL_V*(np.exp(-0.4*x**2.25)-10/(10+x**2.25))
    return Iw2


dd = 0.01
d_arr = np.arange(dd - 1, 1 - dd, dd)

dv = 0.01
v_max = 10.0
v_arr = np.arange(0,v_max,dv)

CL_list = []
wG_list = []
wL_list = []
err_list = []
errW_list = []

##def fitfun(x, CL_fit, wG_fit, wL_fit):
##    return (1-CL_fit) * G(x, wG_fit) + CL_fit * L(x, wL_fit)

def fitfun(x, CL_fit):#, wG_fit, wL_fit):
    return (1-CL_fit) * G(x, 1.0) + CL_fit * L(x, 1.0)


popt = (0.1)#, 0.9, 0.1)
for d in d_arr:
    R = get_R(d)
    wG = (1 - d)/(2*R)
    wL = (1 + d)/(2*R)
    I = V(v_arr, wG, wL)
    
    popt, pcov = curve_fit(fitfun, v_arr, I, p0=popt)
    CL_list.append(popt[0])
##    wG_list.append(popt[1])
##    wL_list.append(popt[2])
    err = np.sum(np.abs(I - fitfun(v_arr, *popt)))*dv
    err_list.append(err)

    errW = np.sum(np.abs(I - approx_V(v_arr, wG, wL)))*dv
    errW_list.append(errW)

##    plt.title('d = {:.2f}'.format(d))
##    plt.xlim(0.0, 5.0)
##    plt.ylim(-0.006,0.011)
##    plt.axhline(0,c='k')
####    plt.plot(v_arr, I)
####    plt.plot(v_arr, fitfun(v_arr, *popt), 'k--')
##    plt.plot(v_arr, I - fitfun(v_arr, *popt))
##    plt.plot(v_arr, I - approx_V(v_arr, wG, wL))
##    plt.show()

##plt.plot(d_arr, err_list)
CL_arr = np.array(CL_list)
##wG_arr = np.array(wG_list)
##wL_arr = np.array(wL_list)

plt.plot(d_arr, CL_arr, label='CL')
plt.plot(d_arr, err_list, label='this work')
plt.plot(d_arr, errW_list, 'k--',label='Whiting2')
##plt.plot(d_arr, wG_arr, label='wG')
##plt.plot(d_arr, wL_arr, label='wL')
##plt.plot(d_arr, approx_R(d_arr), 'k--')
plt.legend()
plt.show()

##
##plt.axhline(0, c='k')
##plt.axvline(0, c='k')
##plt.plot([0,0.5,0.5],[0.5,0.5,0],'k--',lw=1)
##plt.plot(v_arr, G(v_arr, 1.0) / G(0.0, 1.0))
##plt.plot(v_arr, L(v_arr, 1.0) / L(0.0, 1.0))
##plt.plot(v_arr, V_1(v_arr, -0.9999) / V_1(0.0, -0.9999), 'k--')
##plt.show()






    
