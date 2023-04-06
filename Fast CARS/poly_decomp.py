from sympy import *
from sympy.series.formal import compute_fps

t,x,dx,G0,G1,G2,G3,G4,G5 = symbols('t x ∆x G0 G1 G2 G3 G4 G5', real=True)

ex1 = -(t)**3 + 3*(t-1)**3 - 3*(t-2)**3 + (t-3)**3

n = 3

print(ex1.simplify())
for k in range(n+1):
    ex2 = -(-1)**k * binomial(n,k)*(t-k)**n
    print(ex2.expand())
