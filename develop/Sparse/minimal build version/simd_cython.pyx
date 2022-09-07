#cython: language_level=3

#import cython

cimport simd #imports the pxd file
cpdef add_flt(a, b):
    return simd.add_flt(a,b)