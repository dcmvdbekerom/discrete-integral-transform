
import numpy as np
from numpy import pi, sqrt, log, exp, cosh, arccosh
import matplotlib.pyplot as plt


gL  = lambda v,v0,w: 2/(pi*w) * 1 / (1 + 4*((v-v0)/w)**2)
gL2 = lambda v,v0,w: gL(v,v0,w) + gL(v,-v0,w)

def gL_corr(v_max, wL):
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


# Make a grid:
Nv = 10000
dv = 0.001

v_arr = np.arange(-Nv,Nv)*dv
v_max = Nv*dv

# Make a Lorentzian:
wL = 1.0
I = gL(v_arr, 0.0, wL)

# Make the error curve by direct summation:
Is = np.zeros(len(v_arr))
for k in range(1,2000):
    Ik = gL2(v_arr, 2*k*v_max, wL)
    Is += Ik
    
##    I0 = gL2(0.0*v_max, 2*k*v_max, wL)
##    Ih = gL2(0.5*v_max, 2*k*v_max, wL)
##    Ip = gL2(1.0*v_max, 2*k*v_max, wL)
##
##    Bk = (2*Ih**2 - Ip*I0 - I0**2)/(4*Ih - 3*I0 - Ip)
##    Ak = I0 - Bk
##    wk = v_max / arccosh((Ip - Bk) / Ak)
##    I_err = Ak * cosh(v_arr/wk) + Bk

##    plt.plot(v_arr,Ik)
##    plt.plot([0,0.5*v_max,v_max],[I0, Ih, Ip],'ks')
##    plt.plot(v_arr, I_err, 'k--')




# Get the error curve by the procedure:
coeffs, S_arr = gL_corr(v_max, wL)
I_err = coeffs[0] * cosh(v_arr / coeffs[2]) + coeffs[1]


# Plot curves:
plt.axvline(0, c='k', ls='--', lw=1)
plt.axhline(0, c='k', ls='--', lw=1)

plt.plot(v_arr, I)
plt.plot(v_arr, Is)

plt.plot(v_arr, I_err, 'k--')
plt.plot([0, v_max/2, v_max],S_arr,'ks')

plt.show()
