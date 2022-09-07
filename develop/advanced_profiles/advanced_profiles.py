import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
from numpy import pi, sqrt

def K_V(x, y):
    z = x + 1j*y
    res = wofz(z)
    return res.real


def K_SDV(x, y, q):
    X = (y - 1j*x)/q - 3/2
    Y = 1/(4*q**2)
    zp = sqrt(X + Y) + sqrt(Y)
    zm = sqrt(X + Y) - sqrt(Y)
    delta_wofz = wofz(1j*zm) - wofz(1j*zp)
    res = delta_wofz
    return res.real


def K_SDV2(x, y, q):
    ## SDV can be made by regular Voigts, so we can implement this
    ## without much changes; q just changes the G & L widths.

    #TO-DO: both now have the x-axis included; should be only one.
    
    X = (y - 1j*x)/q - 3/2
    Y = 1/(4*q**2)
    zp = sqrt(X + Y) + sqrt(Y)
    zm = sqrt(X + Y) - sqrt(Y)
    res = K_V(-zm.imag, zm.real) - K_V(-zp.imag, zp.real)
    return res



def K_SDV2(x, y, q):
    ## SDV can be made by regular Voigts, so we can implement this
    ## without much changes; q just changes the G & L widths.

    #TO-DO: both now have the x-axis included; should be only one.
    
    X = (y - 1j*x)/q - 3/2
    Y = 1/(4*q**2)
    zp = sqrt(X + Y) + sqrt(Y)
    zm = sqrt(X + Y) - sqrt(Y)


    (y - 1j*x)/q - 3/2 + 1/(4*q**2)

    
    res = K_V(-zm.imag, zm.real) - K_V(-zp.imag, zp.real)
    return res


def K_R(x, y, zeta):
    z = x + 1j*y
    wofz0 = wofz(z)
    res = wofz0 / (1 - sqrt(pi)*zeta*wofz0)
    return res.real


def K_R2(x, y, zeta):
    z = x + 1j*y
    wofz0 = wofz(z).real
    t = sqrt(pi)*zeta*wofz0
    res = wofz0 * (1 + t + t**2 + t**3 + t**4)
    return res


def K_SDR(x, y, q, zeta):
    X = (y + zeta - 1j*x)/q - 3/2
    Y = 1/(4*q**2)
    zp = sqrt(X + Y) + sqrt(Y)
    zm = sqrt(X + Y) - sqrt(Y)
    delta_wofz = wofz(1j*zm) - wofz(1j*zp)
    res = delta_wofz / (1 - sqrt(pi)*zeta*delta_wofz)
    return res.real


x_max = 10.0
dx = 0.001
x_arr = np.arange(0,x_max,dx)
y = 1.0
zeta = 1.0
I_arr = K_R(x_arr, y, zeta)

##plt.plot(x_arr, K_V(x_arr,y), label='q=0.0')




##zeta_arr = np.linspace(0,1,101)[1:]
##
##y = [K_R(0,1,z) for z in zeta_arr]
##
##plt.plot(zeta_arr, y)





##for zeta in [0.1, 0.2, 0.5, 1.0]:
##    I_arr = K_R(x_arr, y, zeta)
##    plt.plot(x_arr, I_arr, label=f'zeta={zeta:.1f}')
##
##    I2_arr = K_R2(x_arr, y, zeta)
##    plt.plot(x_arr, I2_arr, 'k--')


plt.plot(x_arr, K_V(x_arr,y), label='q=0.0')
for q in [0.1, 0.2, 0.5, 1.0]:
    I_arr = K_SDV(x_arr, y, q)
    plt.plot(x_arr, I_arr, label=f'q={q:.1f}')

    I2_arr = K_SDV2(x_arr, y, q)
    plt.plot(x_arr, I2_arr, 'k--')

plt.legend()
plt.show()

