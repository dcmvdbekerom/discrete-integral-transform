import numpy as np
new = np.newaxis

a = np.array([1,2])
b = np.array([3,4,5])
c = np.array([200,300])

d = a[new,new,:]+b[new,:,new]+c[:,new,new]
print(d)
