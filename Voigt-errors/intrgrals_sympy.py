from sympy import *
from sympy.integrals.transforms import mellin_transform
from sympy.functions.special.error_functions import erfc
from sympy.printing.mathml import mathml
import numpy as np


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

erfcx = sqrt(pi)*erfc(b/a)*exp(b**2/a**2)

c_list = []
def intfun(f):
    global c_list
    res = 2*integrate(f,(t,0,oo))*a
    res = res.expand()
    res = res.collect(erfcx)

    evens  = res.coeff(erfcx,1).subs(b/a,x)
    odds   = res.coeff(erfcx,0).subs(b/a,x)

    print(evens)
    print(odds)
    print('')
    
    powers = evens + odds

    c_list.append([powers.coeff(x,i) for i in range(9)])
  
    return res/a

L1L1 = intfun(GG0**2 * GL1**2)
L1L2 = intfun(GG0**2 * GL1*GL2)
L2L2 = intfun(GG0**2 * GL2**2)

G1L1 = intfun(GG0*GL0 * GG1*GL1)
G1L2 = intfun(GG0*GL0 * GG1*GL2)
G2L1 = intfun(GG0*GL0 * GG2*GL1)
G2L2 = intfun(GG0*GL0 * GG2*GL2)

G1G1 = intfun(GL0**2 * GG1**2)
G1G2 = intfun(GL0**2 * GG1*GG2)
G2G2 = intfun(GL0**2 * GG2**2)

c_arr = np.array(c_list)
print(c_arr)

S_GG = G1L1*G2L1 - G1G2*L1L1
S_GL = G1L1*L1L2 - G1L2*L1L1
S_LL = G1L1*G1L2 - G1G1*L1L2
S_LG = G1G2*G1L1 - G1G1*G2L1
S_V  = G1G1*L1L1 - G1L1**2

