from sympy import *

t,a,v,dv,vk,Dv = symbols('t a v dv vk Dv',positive=True,real=True)
init_printing(use_unicode=True)

print('Weights:')

Nbin = 2 # Number of bins
n_min = -((Nbin - 1) // 2)
n_max = n_min + Nbin - 1

eqs = []
for k in range(Nbin):  # order in Taylor series
    eq = 0
    for i in range(n_min, n_max + 1): # bin index
        eq += Indexed('a',i)*(t-i)**k
    eqs.append(eq)
        
eqs[0] -= 1

a_n = [Indexed('a',k) for k in range(n_min, n_max + 1)]
res = solve(eqs,a_n)

for a in res:
    print(a, res[a])
print('')


print('Derivatives:')

Ndiff = 0 # N'th derivative
i_min = -(Ndiff  // 2)

veq = solve((v-vk)/Dv-t,v)[0]
for a in res:
    eqa = res[a]
    eq = 0
    for i in range(Ndiff + 1):
        i_bin = i_min + i
        coeff = (-1)**(i + Ndiff) * binomial(Ndiff, i)
        eq += coeff * eqa.subs(t, (v + i_bin * dv - vk)/Dv) / dv**Ndiff
        
        if a == a_n[0]:
            print(i_bin,':', coeff)
            
    if a == a_n[0]:
        print('')

    eq = eq.simplify().subs(v,veq).simplify()
    print(a, eq)
