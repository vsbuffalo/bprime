import numpy as np
from setuptools import setup, Extension

extra_compile_args = ['-O3', '-Wall']

likclib_ext = Extension('likclib',
                        extra_compile_args=extra_compile_args,
                              include_dirs=[np.get_include()],
                              language='c',
                        sources = ['bgspy/src/likelihood.c'])

setup(
    name='bgspy',
    version='0.1dev',
    packages=['bgspy',],
    license='BSD',
    long_description=open('README.md').read(),
    #scripts=[],
    entry_points = {
        'console_scripts': ['bgspy=bgspy.command_line:cli']
    },
    ext_modules=[likclib_ext],
)
