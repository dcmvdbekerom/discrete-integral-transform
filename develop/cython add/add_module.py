from cython_add import cython_add_at
from numpy import add

def numpy_add_at(LDM, k, l, m, I):
    return add.at(LDM,(k,l,m),I)

