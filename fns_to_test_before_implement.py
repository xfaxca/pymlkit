import numpy as np
import pandas as pd
class FeatureGenerator(object):
    """
    Class containing several methods for the generation of new features via the transformation
        existing features that are specified by the user. Each feature used as input will be combined
        in a specific manner (depending on the method used) with each other feature and itself, generating
        n^2 new features in addition to the n features passed.
    """
    def __init__(self, name='FeatureGenerator'):
        self.name = name
        self._version = '0.1'
        # add more attributes as needed

    def fit_sq(self, X, y=None):
        """
        Generates n^2 new features that are combinations of the square of each feature couple.
        :return:
        """
        if isinstance(X, pd.DataFrame):
            print('Features passed in dataframe --> ', X.columns.values)
            pass
        elif isinstance(X, np.ndarray):
            print('%i features passed as ndarray to FeatureGenerator.fit()' % len(X))
            pass
        self.generated = []
        pass
        # insert generated features into a self.generated variable and then return this in the fit_transform method


    def fit_transform_sq(self, X, y):
        return self.fit_sq(X=X)
