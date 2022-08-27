from setuptools import Extension, setup
#from Cython.Build import cythonize
from numpy import get_include
import sys

def get_ext_modules(with_binaries):
    """
    Parameters
    ----------
    with_binaries: bool
        if False, do not try to build Cython extensions
    """
    # Based on code from https://github.com/pallets/markupsafe/blob/main/setup.py

    print(sys.version)
    ext_modules = []
    cmdclass = {}

    if not with_binaries:
        # skip building extensions
        return {"cmdclass": cmdclass, "ext_modules": ext_modules}
    try:
        import cython
    except (ModuleNotFoundError) as err:
        raise BuildFailed(
            "Cython not found : Skipping all Cython extensions...!"
        ) from err

    print("Cython " + cython.__version__)

    from Cython.Distutils import build_ext

    class build_ext_subclass(build_ext):
        def build_extensions(self):
            c = self.compiler.compiler_type
            copt = {
                "msvc": ["/openmp", "/Ox", "/fp:fast", "/favor:INTEL64"],
                "mingw32": ["-fopenmp", "-O3", "-ffast-math", "-march=native"],
            }

            lopt = {"mingw32": ["-fopenmp"]}

            print("Compiling with " + c + "...")
            try:
                for e in self.extensions:
                    e.extra_compile_args = copt[c]
            except (KeyError):
                pass
            try:
                for e in self.extensions:
                    e.extra_link_args = lopt[c]
            except (KeyError):
                pass
            try:
                build_ext.build_extensions(self)
            except (CCompilerError, DistutilsExecError, DistutilsPlatformError) as err:
                raise BuildFailed() from err

    ext_modules.append(
        Extension(
            "py_simd_cpp",
            sources=[
                "simd_cython.pyx",
                "simd.cpp", #must be included as well
            ],
            include_dirs=[get_include()],
            language="c++",
            extra_link_args=[],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    )

    cmdclass["build_ext"] = build_ext_subclass

    return {"cmdclass": cmdclass, "ext_modules": ext_modules}





setup(
    **get_ext_modules(True),
    #ext_modules = cythonize("cpp_simd.pyx"),
    include_dirs = [get_include()],
    
)

