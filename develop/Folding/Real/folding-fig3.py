import numpy as np
from numpy import pi, log, exp, sqrt, cosh, sinh, arccosh
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

## function definitions:

coth = lambda x: cosh(x)/sinh(x)
csch = lambda x: 1/sinh(x)

gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gG  = lambda v,v0,w: (2/w)*sqrt(log(2)/pi)*exp(-4*log(2)*((v-v0)/w)**2)
gE = lambda v,v0,w: exp(-np.abs(v-v0)/w)/(2*w)
gE_corr = lambda v,w,vmax: (exp((v-2*vmax)/w) + exp(-v/w))/(1-exp(-2*vmax/w))/(2*w)
dgLdv = lambda v,v0,w: -32/(pi*w**2) * ((v-v0)/w)/(1+4*((v-v0)/w)**2)**2

gL_FT = lambda x,w: exp(-np.abs(x)*pi*w)
gG_FT = lambda x,w: exp(-(x*pi*w)**2/(4*log(2)))
gV_FT = lambda x,wG,wL: gG_FT(x,wG)*gL_FT(x,wL)
gE_FT = lambda x,w: 1 / (1 + 4*pi**2*x**2*w**2)


def gL_err(v_max, wL):
    #We calculate the value of three points on the
    #error curve: v = [0, vmax/2, vmax]; These
    #points will later be used as anchor points
    #for a cosh(v) approximation of the error curves.

    #By approximating the Lorentzian by a simple inverse parabola (1/x**2),
    #Values of the infinite sum can be calculate exactly:
    S0 = wL/v_max**2 *  pi/24
    Sh = wL/v_max**2 * (pi/4 - 2/pi)
    Sp = wL/v_max**2 * (pi/8 - 1/(2*pi))

    #The approximation is a bit rough for low k.
    #here we calculate the first few by hand
    #and subtract the approximated part for each k
    S_arr = np.array([S0, Sh, Sp])
    delta = np.array([0.0, 0.5, 1.0])

    for k in range(1,5):
        dp = 2*k + delta
        dm = 2*k - delta

        S_arr -= 2/(pi*wL)*(
              1/((4+16*(dm*v_max/wL)**2)*(dm*v_max/wL)**2)
            + 1/((4+16*(dp*v_max/wL)**2)*(dp*v_max/wL)**2))

    S0, Sh, Sp = S_arr

    #calculate coefficients for the error curve approximation:
    Bs = (2*Sh**2 - Sp*S0 - S0**2)/(4*Sh - 3*S0 - Sp)
    As = S0 - Bs
    ws = v_max / arccosh((Sp - Bs) / As)

    return [As,Bs,ws], S_arr


##def calc_gV_FT(x, wG, wL, folding_thresh=1e-6):
##
##    gV_FT = lambda x,wG,wL: gG_FT(x,wG)*gL_FT(x,wL)
##
##    result = np.zeros(x.size)
##    n = 0
##    while gV_FT(n/2,wG,wL) >= folding_thresh:
##        result += gV_FT(n/2 + x[::1-2*(n&1)], wG, wL)
##        n += 1
##
##    return result

coeff_w = [0.39560962,-0.19461568]
coeff_A = [0.09432246, 0.06592025]
coeff_B = [0.11202818, 0.09048447]

corr_fun = lambda x,c0,c1: c0 * np.exp(-c1*x**2)

def gL_FT_corr(x_arr, wL):

    result = gL_FT(x_arr, wL)

    vmax = 1/(2*x_arr[1])
    q = wL/vmax
    
    w_corr = corr_fun(q, *coeff_w)*vmax
    A_corr = corr_fun(q, *coeff_A)*q
    B_corr = corr_fun(q, *coeff_B)*q

    I_corr = A_corr * gE_FT(x_arr, w_corr)
    I_corr[0] += 2*B_corr
    I_corr[1::2] *= -1

    result -= I_corr
    return result


## produce lineshapes:


Nv = 1000
dv = 0.02

v_arr = np.arange(-Nv,Nv)*dv
v_max = Nv*dv

x_arr = np.fft.rfftfreq(2*Nv, dv)
dx = x_arr[1]
##wG = 0.0
wL = 1.0

# The lineshape we get when we calculate in real space:
I_direct = gL(v_arr,0,wL)
I_direct_FT = np.fft.rfft(np.fft.fftshift(I_direct*dv))

window = (v_arr>=-v_max/2)&(v_arr<v_max/2)
I_windowed = I_direct * window
I_windowed_FT = np.fft.rfft(np.fft.fftshift(I_windowed*dv))



c,S = gL_err(v_max,wL)
I_err_ex = c[0] * cosh(v_arr/c[2]) + c[1]


fig, ax = plt.subplots(1,2)

ax[0].set_title('Real space')
ax[0].axhline(0,c='k',ls='--')
ax[0].axvline(0,c='k',ls='--')
ax[0].plot(v_arr, I_direct, 'r', lw=1, label = 'Direct')
ax[0].plot(v_arr, I_direct-I_windowed, 'b--', lw=1, label = 'Windowed')


ax[1].set_title('FT space')
ax[1].axhline(0,c='k',ls='--')
ax[1].axvline(0,c='k',ls='--')

ax[1].plot(x_arr, I_direct_FT, 'r-', lw=1, label='FT Direct')
ax[1].plot(x_arr, I_direct_FT-I_windowed_FT, 'b.-', lw=1, label='FT Windowed')
ax[1].plot(x_arr, np.sinc(x_arr/(2*dx)), '-', lw=1)
##ax[1].set_yscale('log')
plt.xlabel('x / (2*v_max)')
ax[1].legend()
plt.savefig('fig3.png')
plt.show()
