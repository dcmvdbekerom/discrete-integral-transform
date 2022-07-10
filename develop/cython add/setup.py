from setuptools import Extension, setup
from Cython.Distutils import build_ext
from numpy import __path__ as npypath

copt =  {'msvc': ['/openmp', '/Ox', '/fp:fast','/favor:INTEL64']  ,
     'mingw32' : ['-fopenmp','-O3','-ffast-math','-march=native']}
lopt =  {'mingw32' : ['-fopenmp'] }

class build_ext_subclass(build_ext):
    def build_extensions(self):
##        print('THIS IS A CUSTOM BUILDEXTSUBCLASS')
##        c = self.compiler.compiler_type
##        try:
##           for e in self.extensions:
##               e.extra_compile_args = copt[c]
##        except(KeyError):
##            pass
##        try:
##            for e in self.extensions:
##                e.extra_link_args = lopt[c]
##        except(KeyError):
##            pass
        build_ext.build_extensions(self)

##mod = Extension('_wripaca',
##            sources=['../wripaca_wrap.c', 
##                     '../../src/wripaca.c'],
##            include_dirs=['../../include']
##            )
##
##setup (name = 'wripaca',
##   ext_modules = [mod],
##   py_modules = ["wripaca"],
##   cmdclass = {'build_ext': build_ext_subclass } )



ext_modules = [Extension('cython_add',
                   sources=['cython_add.pyx'],
                   include_dirs=[npypath[0]+'/core/include'],
                   language='c',
                   #extra_compile_args=[
                   #    '/O2','/favor:blend','/fp:fast'],
                   extra_link_args=[],
                   )]
setup(name = 'cython_extra_spicy', 
      cmdclass = {'build_ext': build_ext_subclass},
      ext_modules = ext_modules)
