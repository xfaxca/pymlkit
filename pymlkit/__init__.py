# __init__.py
"""
pymlkit
-------------
        Toolkit for the creation of machine learning algorithms in Python. Contains functions
    that serve as wrappers to scikit-learn and mlxtend as well as custom modules for data cleaning,
    pre-processing, model selection, estimator tuning and model evaluation.

Submodules
-------------
explore
        Module containing functionality for exploratory data analysis and visualization.

preprocessing
        Module containing functionality for data cleansing, imputation, feature mapping and scaling.

model_select
        Module containing functionality for selecting models, including cross validation checks, model scans for
    common classifiers/regressors, creation of ensemble models (bagging, majority vote, etc.)
"""

__version__ = "pymlkit-v0.0.2"
__all__ = ['preprocessing',
           'explore',
           'model_selection']
