# __init__.py

import pymlkit.preprocessing
from pymlkit.preprocessing import *
import pymlkit.explore
from pymlkit.explore import *

__doc__ = "Package: pymlkit\n" \
          "Created: 11/01/2016\n" \
          "Author contact: cameron@tutanota.com \n" \
          "Description: Toolkit for the creation of machine learning algorithms in Python. Contains functions \n" \
          "that serve as wrappers to scikit-learn and mlxtend as well as custom modules for data cleaning, \n" \
          "pre-processing, model selection, estimator tuning and model evaluation."
__author__ = "Cameron Faxon"
__copyright__ = "Copyright (C) 2016 Cameron Faxon"
__license__ = "GNU GPLv3"
__version__ = "0.0.1"
__all__ = ['class_proportions', 'see_nulls',
           'yn_binarize', 'map_feature', 'impute_by_other', 'impute_all', 'DataFrameImputer',
           'strat_shufflesplit', 'random_undersample', 'random_oversample', 'balance_classes_adasyn',
           'balance_classes_smote']

