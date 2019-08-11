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
      install_requires=[['xgboost==0.90',
                         'pandas==0.23.4',
                         'nltk==3.3',
                         'seaborn==0.9.0',
                         'matplotlib==2.2.3',
                         'numpy==1.16.4',
                         'gensim==3.4.0',
                         'imblearn==0.0',
                         'scikit_learn==0.21.3',
                         'textblob==0.15.3']])
