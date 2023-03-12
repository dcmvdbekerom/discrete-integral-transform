from sympy import *
from sympy.series.formal import compute_fps

t,x,dx,G0,G1,G2,G3 = symbols('t x ∆x G0 G1 G2 G3', real=True)

##ex1 = fps(exp(x), x).truncate(3)


tt = [t+1, t, t-1]
a  = [t*(t-1)/2, 1-t**2, t*(t+1)/2]

exp0 = 1 + x + x**2/2 + x**3/6
exp = [exp0.subs(x, ti*dx) for ti in tt]

dGdlnx = [G0, x*G1, x*G1 + x**2*G2, x*G1 + 3*x**2*G2 + x**3*G3]
rfact = [1, 1, 1/2, 1/6]


for n in [0,1,2,3]:
    print(n)
    err = rfact[n]*(  a[0]*exp[0]*tt[0]**n
                    + a[1]*exp[1]*tt[1]**n
                    + a[2]*exp[2]*tt[2]**n)*dGdlnx[n]*dx**n

    print(err.simplify())
    print('')
