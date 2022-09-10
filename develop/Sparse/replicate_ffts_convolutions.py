import numpy as np
import matplotlib.pyplot as plt
from numpy import sinc, pi, exp, sin
from numpy.fft import rfft, rfftfreq, fftshift, ifftshift

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


def rect(v,w):
    return 1.0*(-0.5 <= v/w)*(v/w < 0.5)
    #return 0.5*(np.abs(v_arr/w) < 0.5) + 0.5*(np.abs(v_arr/w) <= 0.5)

def rect2(v,M):
    res = np.zeros(N)
    res[(N//2)-(M//2):(N//2)+(M//2)+(M&1)] = 1.0
    return res
    #return 0.5*(np.abs(v_arr/w) < 0.5) + 0.5*(np.abs(v_arr/w) <= 0.5)


w = 1.0
M = 11

y_arr = np.fft.fftshift(rect(v_arr,w)/w)
y_arr2 = np.fft.fftshift(rect2(v_arr,M))
Y_rfft = rfft(y_arr2)*dv
Y_exact = sinc(w*x_arr)
Y_myft = myft(y_arr2)*dv
##Y_rect = np.sum([exp(-2j*pi*k*n_arr/N) for k in np.arange(-(M//2),(M//2)+(M&1))],0) * dv
##Y_rect = np.sum([exp(-2j*pi*(k - M//2)*n_arr/N) for k in np.arange(M)],0) * dv
##Y_rect = (1-exp(-2j*pi*M*n_arr/N)) / (1-exp(-2j*pi*n_arr/N)) * exp(2j*pi*(M//2)*n_arr/N) * dv
Y_rect = sin(pi*M*n_arr/N)/sin(pi*n_arr/N) * dv

fig, ax = plt.subplots(1,2, figsize=(14,6))
ax[0].plot(np.fft.fftshift(k_arr), y_arr2,'k.')
ax[1].plot(n_arr, Y_rfft.real,'b')
ax[1].plot(n_arr, Y_rfft.imag,'r')
##ax[1].plot(n_arr, Y_myft.real, 'k--')
##ax[1].plot(n_arr, Y_myft.imag, 'k--')
ax[1].plot(n_arr, Y_rect.real, 'k--')
ax[1].plot(n_arr, Y_rect.imag, 'k--')
plt.show()
