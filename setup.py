from distutils.core import setup

setup(
    name='bgspy',
    version='0.1dev',
    packages=['bgspy',],
    license='BSD',
    long_description=open('README.md').read(),
    scripts=['tools/fit_sims.py'],
    entry_points = {
        'console_scripts': ['bgspy=bgspy.command_line:main'],
    }
)
