from sympy import *
from sympy.integrals.transforms import mellin_transform
from sympy.functions.special.error_functions import erfc
from sympy.functions.combinatorial.factorials import factorial, factorial2
from sympy.printing.mathml import mathml
import numpy as np
import sys

t,a,b,x = symbols('t a b x',positive=True,real=True)
##n = symbols('n',integer=True)
##erfcx = Function('erfcx')
init_printing()

# a = pi*aG/(2*log(2))**0.5
# b = pi*aL

GG0 = exp(-a**2*t**2/2)
GG1 = -a**2*t**2*GG0
GG2 =  a**2*t**2*(a**2*t**2-2)*GG0

GL0 = exp(-b*t)
GL1 = -b*t*GL0
GL2 = b*t*(b*t-1)*GL0

GV0 = GG0 * GL0

erfcx  = sqrt(pi)*erfc(b/a)*exp(b**2/a**2)
erfcxx = sqrt(pi)*erfc(x)*exp(x**2)

names = ['L1L1','L1L2','L2L2','G1L1','G1L2','G2L1','G2L2','G1G1','G1G2','G2G2']
name_i = 0


erf_poly = 0
for n in range(20):
    erf_poly += (-1)**n*x**(2*n+1)/(factorial(n)*(2*n+1))
erf_poly *= 2/sqrt(pi)


exp2_poly = 0
for n in range(10):
    exp2_poly += x**(2*n)/factorial(n)

erfcx_poly = ((1-erf_poly)*exp2_poly).expand()
erfcx_poly += O(x**11)

print(erfcx_poly)



c_list = []
def intfun(f):
    global c_list,name_i
    print(name_i,names[name_i])
    name_i += 1
    
    res = 2*integrate(f,(t,0,oo))*a
    res = res.expand()
    res = res.collect(erfcx)

    evens  = res.coeff(erfcx,1).subs(b/a,x)
    odds   = res.coeff(erfcx,0).subs(b/a,x)

    powers = evens + odds
    c = [powers.coeff(x,i) for i in range(9)]
    c_list.append(c)

    approx_h = 0
    for n in range(8,-8,-2):
        if n>0:
            approx_h += c[n-1]*x**n
        for i in range(max(n,0),10,2):
            m = (i-n)//2
            g = factorial2(2*m-1)/(-2)**m
            approx_h += c[i]*g*x**n
    approx_h *= 2/pi

    approx_l = 0
    for n in range(0,10,2):
        approx_l += c[n]*pi**0.5*erfcx_poly*x**n
        if n > 0:
            approx_l += c[n-1]*x**(n-1)
    approx_l *= sqrt(2*log(2))/pi
    approx_l = approx_l.factor().simplify()
    
    print(approx_h.expand())
    print('')
  
    return res/a,approx_h,approx_l

# Calculate error integrals.
# 'h' are asymptotes when alpha -> oo
# 'l' are asymptotes when alpha -> 0

L1L1,L1L1_h,L1L1_l = intfun(GG0**2 * GL1**2)
L1L2,L1L2_h,L1L2_l = intfun(GG0**2 * GL1*GL2)
L2L2,L2L2_h,L2L2_l = intfun(GG0**2 * GL2**2)

G1L1,G1L1_h,G1L1_l = intfun(GG0*GL0 * GG1*GL1)
G1L2,G1L2_h,G1L2_l = intfun(GG0*GL0 * GG1*GL2)
G2L1,G2L1_h,G2L1_l = intfun(GG0*GL0 * GG2*GL1)
G2L2,G2L2_h,G2L2_l = intfun(GG0*GL0 * GG2*GL2)

G1G1,G1G1_h,G1G1_l = intfun(GL0**2 * GG1**2)
G1G2,G1G2_h,G1G2_l = intfun(GL0**2 * GG1*GG2)
G2G2,G2G2_h,G2G2_l = intfun(GL0**2 * GG2**2)

c_arr = np.array(c_list)
print(c_arr)

R_GG_h = ((G1L1_h*G2L1_h - G1G2_h*L1L1_h).expand()/(G1G1_h*L1L1_h - G1L1_h**2).expand()).cancel(x)
R_LG_h = ((G1G2_h*G1L1_h - G1G1_h*G2L1_h).expand()/(G1G1_h*L1L1_h - G1L1_h**2).expand()).cancel(x)

R_GG_l = ((G1L1_l*G2L1_l - G1G2_l*L1L1_l)/(G1G1_l*L1L1_l - G1L1_l**2)).factor().simplify().factor()
R_LG_l = ((G1G2_l*G1L1_l - G1G1_l*G2L1_l)/(G1G1_l*L1L1_l - G1L1_l**2)).factor().simplify().factor()
