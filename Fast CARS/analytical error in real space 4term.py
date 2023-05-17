from sympy import *
from sympy.series.formal import compute_fps

t,x,dx,G0,G1,G2,G3,G4,G5 = symbols('t x ∆x G0 G1 G2 G3 G4 G5', real=True)

tt = [t+1, t, t-1, t-2]
a  = [
t*(-t**2 + 3*t - 2)/6,
(t**3 - 2*t**2 - t + 2)/2,
t*(-t**2 + t + 2)/2,
t*(t**2-1)/6,
]

exp0 = 1 + x + x**2/2 + x**3/6 + x**4/24
exp = [exp0.subs(x, ti*dx) for ti in tt]

ex2 = 6*(-t**3 +   t**2 + 2*t    )*(t**2 - 2*t + 1)**2 * dx**4 -   ( t**3 - 3*t**2 + 2*t    )*(t**2 + 2*t + 1)**2 * dx**4+ 3*( t**3 - 2*t**2 -   t + 2)* t**4               * dx**4 

dGdlnx = [  G0,
          x*G1,
          x*G1 + x**2*G2,
          x*G1 + 3*x**2*G2 +   x**3*G3,
          x*G1 + 7*x**2*G2 + 6*x**3*G3 + x**4*G4,
          x*G1 + 15*x**2*G2 + 25*x**3*G3 + 10*x**4*G4 + x**5*G5,
          ]


G0 + 15*x**1*G1 + 31*x**2*G2 + 10*x**3*G3 + x**4*G4

rfact = [1, 1, 1/2, 1/6, 1/24]


for n in [0,1,2,3,4]:
    print(n)
    err = rfact[n]*(  a[0]*exp[0]*tt[0]**n
                    + a[1]*exp[1]*tt[1]**n
                    + a[2]*exp[2]*tt[2]**n
                    + a[2]*exp[2]*tt[2]**n)*dGdlnx[n]*dx**n

    print(err.simplify())
    print('')
