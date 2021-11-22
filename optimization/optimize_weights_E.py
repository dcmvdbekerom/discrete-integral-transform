from sympy import *

t,dxE = symbols('t dxE',positive=True,real=True)
a = symbols('a',real=True)

init_printing(use_unicode=True)

##t = 0.25
c1 = (t - a)*dxE
c2 = (t**2 - 2*a*t + a)*dxE**2/2

E = (
        c1**2/2 +
       -c1*c2/2 +
        c2**2/4 +
        
        0)

dEda = diff(E,a).factor()

res = solve(dEda,a)[0]
print(res)
