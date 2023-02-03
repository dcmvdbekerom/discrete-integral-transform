import numpy as np
import matplotlib.pyplot as plt
from numpy import sinc, pi, exp, sin, cos, sinh, cosh
from numpy.fft import rfft, rfftfreq, fftshift, ifftshift
from scipy.special import hyp2f1

Nv = 100
dv = 0.1

k_arr = np.arange(-Nv, Nv)
v_arr = k_arr*dv
v_max = Nv*dv

n_arr = np.arange(0, Nv + 1)
x_max = 1/(2*dv)
dx = 1/(2*v_max)
x_arr = n_arr * dx

N = 2*Nv

def myft(y_arr):
    return np.sum([y_arr[k]*exp(-2j*pi*k*n_arr/N) for k in k_arr],0)


def hyper_sum(a,z,m):
    return hyp2f1(a, 1, a + 1, z) - a*z**m/(m + a) * hyp2f1(m + a, 1, m + a + 1, z)
    

def direct_L_FT():

    z = exp(-2j*pi*n_arr/N)
    z[np.abs(z)==0] = 0.0
    a = w/(2*dv)
    
    return 2/(pi*w) * np.sum(
      [z**  k /(1 + (k/a)**2) for k in range(Nv)]
    + [z**(-k)/(1 + (k/a)**2) for k in range(Nv)]
    +[(-1)**n_arr/(1 + 4*(Nv*dv/w)**2) - 1]
    ,0)


def direct_L_FT2():

    z = exp(-2j*pi*n_arr/N)
    a = w/(2*dv)
    
    return 2/(pi*w) * (
      0.5*(hyper_sum(1j*a,  z,Nv) + hyper_sum(-1j*a,  z,Nv))
    + 0.5*(hyper_sum(1j*a,1/z,Nv) + hyper_sum(-1j*a,1/z,Nv))
    +(-1)**n_arr/(1 + 4*(Nv*dv/w)**2) - 1
    )

gL  = lambda v,w: 2/(pi*w) * 1 / (1 + 4*(v/w)**2)


w = 1.0
M = 11

y_arr = np.fft.fftshift(gL(v_arr,w))
Y_rfft = rfft(y_arr)*dv
Y_myft = myft(y_arr)*dv
Y_lor = 2/(pi*w) * np.sum([
    exp(-2j*pi*k*n_arr/N)/(1 + 4*(k*dv/w)**2)
    for k in k_arr],0) * dv

Y_lor2 = 2/(pi*w) * np.sum(
    [2*cos(-2*pi*k*n_arr/N)/(1 + 4*(k*dv/w)**2) for k in range(Nv)]
    +[(-1)**n_arr/(1 + 4*(Nv*dv/w)**2) - 1]
    ,0) * dv

Y_lor3 = direct_L_FT2() * dv

fig, ax = plt.subplots(1,2, figsize=(14,6))
ax[0].plot(np.fft.fftshift(k_arr), y_arr,'k.')
ax[1].plot(n_arr, Y_rfft.real,'b')
ax[1].plot(n_arr, Y_rfft.imag,'r')
##ax[1].plot(n_arr, Y_myft.real, 'k--')
##ax[1].plot(n_arr, Y_myft.imag, 'k--')
ax[1].plot(n_arr, Y_lor3.real, 'k--')
ax[1].plot(n_arr, Y_lor3.imag, 'k--')
plt.show()
