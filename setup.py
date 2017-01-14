# setup.py

from distutils.core import setup

setup(name='pymlkit',
      version='0.1',
      description='Python tools for pre-processing data and building and evaluating machine learning models',
      author='Cameron Faxon',
      license='GNU GPLv3',
      author_email='Cameron@tutanota.com',
      url='https://github.com/xfaxca/pymlkit',
      packages=['pymlkit'],
      requires=['sklearn', 'numpy', 'pandas', 'seaborn', 'matplotlib', 'imblearn'])
