# setup.py

from setuptools import setup, find_packages

setup(name='pymlkit',
      version='v0.0.2',
      description='Python tools for pre-processing data and building and evaluating machine learning models',
      author='Cameron Faxon',
      author_email='xfaxca@tutanota.com',
      license='GNU GPLv3',
      url='https://github.com/xfaxca/pymlkit',
      packages=find_packages(),
      install_requires=['matplotlib==1.5.1',
                        'nltk==3.4.5',
                        'numpy==1.11.2',
                        'seaborn==0.7.1',
                        'scikit_learn==0.18.1',
                        'imbalanced-learn==0.2.1'])
