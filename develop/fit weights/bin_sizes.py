
dv = 0.2
v0 = 13.76

i = v0 / dv
i0 = int(i)
t = i - i0
print(2, i0, t)
print('')

print(i)
for n in range(2,10):
    i = v0 / dv
    i0 = int(i - n/2 + 1)
    t = (i - i0) / (n - 1) #This is the Bezier parameter, not grid alignment
    print(n, i0, t)
    




