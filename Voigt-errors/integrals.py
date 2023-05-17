import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf,erfc,erfcx
from numpy import log,pi,exp,sqrt

f_GG0 = lambda x: exp(-(pi*x)**2/(4*log(2)))
f_GL0 = lambda x: exp(-np.abs(pi*x))

C1 = 1.06920
C2 = 0.86639

f_aG  = lambda d: 2*(1-d)/(C1*(1+d)+(C2*(1+d)**2+4*(1-d)**2)**0.5)
f_aL  = lambda d: 2*(1+d)/(C1*(1+d)+(C2*(1+d)**2+4*(1-d)**2)**0.5)

f_GV0 = lambda x,d: f_GG0(x*f_aG(d)) * f_GL0(x*f_aL(d))

dx = 0.001
x_max = 10.0
x = np.arange(0,x_max,dx)



def Iex(d,n=0):

    a = pi*f_aG(d)/(2*log(2))**0.5
    b = pi*f_aL(d)

    X = (1+d)/(1-d)*(2*log(2))**0.5
    
    
    A = 1.98
    B = 1.135

##    erfcx = lambda x: (1-exp(-A*x))/(B*pi**0.5*x)
    
    I = [
        sqrt(pi)*erfcx(X)/a,
(a - sqrt(pi)*b*erfcx(X))/a**3,
(a**3 - (a - sqrt(pi)*b*erfcx(X))*(a**2 + 2*b**2))/(2*a**5*b),
(3*a**5 - (3*a**2 + 2*b**2)*(a*(a**2 - 2*b**2) + 2*sqrt(pi)*b**3*erfcx(X)))/(4*a**7*b**2),
(-3*a**5*(a**2 + 2*b**2) + (a*(a**2 - 2*b**2) + 2*sqrt(pi)*b**3*erfcx(X))*(3*a**4 + 12*a**2*b**2 + 4*b**4))/(8*a**9*b**3),
(-15*a**7*(3*a**2 + 2*b**2) + (15*a**4 + 20*a**2*b**2 + 4*b**4)*(3*a**5 - 2*a**3*b**2 + 4*a*b**4 - 4*sqrt(pi)*b**5*erfcx(X)))/(16*a**11*b**4),
(15*a**7*(3*a**4 + 16*a**2*b**2 + 4*b**4) - (3*a**5 - 2*a**3*b**2 + 4*a*b**4 - 4*sqrt(pi)*b**5*erfcx(X))*(15*a**6 + 90*a**4*b**2 + 60*a**2*b**4 + 8*b**6))/(32*a**13*b**5),
(105*a**9*(15*a**4 + 24*a**2*b**2 + 4*b**4) - (105*a**6 + 210*a**4*b**2 + 84*a**2*b**4 + 8*b**6)*(15*a**7 - 6*a**5*b**2 + 4*a**3*b**4 - 8*a*b**6 + 8*sqrt(pi)*b**7*erfcx(X)))/(64*a**15*b**6),
(-105*a**9*(15*a**6 + 114*a**4*b**2 + 76*a**2*b**4 + 8*b**6) + (15*a**7 - 6*a**5*b**2 + 4*a**3*b**4 - 8*a*b**6 + 8*sqrt(pi)*b**7*erfcx(X))*(105*a**8 + 840*a**6*b**2 + 840*a**4*b**4 + 224*a**2*b**6 + 16*b**8))/(128*a**17*b**7),
][n]

    return I


d_arr  = np.linspace(-1.0,1.0,51)

##for n in [0,1,2,3,4,5,6,7,8]:
##    xnGV2 = np.array([2*np.sum(x**n * f_GV0(x,d)**2)*dx for d in d_arr])
##    plt.plot(d_arr,xnGV2,label=n)
##    plt.plot(d_arr,Iex(d_arr,n),'k--')
##    
##plt.plot(d_arr,f_aL(d_arr)/(f_aG(d_arr)/(2*log(2))**0.5))
##plt.plot(d_arr,(1+d_arr)/(1-d_arr)*(2*log(2))**0.5,'--')
plt.plot(d_arr,erfcx((1+d_arr)/(1-d_arr)*(2*log(2))**0.5))
    
plt.ylim(0,0.025)
plt.legend()
plt.show()
