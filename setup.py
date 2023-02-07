from setuptools import setup, Extension
import numpy as np

extra_compile_args = ['-O3', '-Wall']


likclib_ext = Extension('likclib',
                        extra_compile_args=extra_compile_args,
                        include_dirs=[np.get_include()],
                        language='c',
                        sources=['bgspy/src/likelihood.c'])


Bclib_ext = Extension('Bclib',
                      extra_compile_args=extra_compile_args,
                      include_dirs=[np.get_include()],
                      language='c',
                      # libraries = ['gsl'],
                      sources=['bgspy/src/theory.c'])


setup(
        name="bgspy",
        version="0.01",
        ext_modules=[likclib_ext, Bclib_ext],
        )
