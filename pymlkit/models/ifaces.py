"""
Module for classes/functions handling model interfaces/compatibility between different libraries.
"""
import xgboost as xgb

__all__ = [
    'XgbWrapper',
    'SklearnWrapper'
]


class XgbWrapper(object):
    """
    Summary: Wrapper for estimator objects from the XGBoost package. Wraps class to provide access to
        uniform fit/train/predict methods. Derived from original code goes to user Eliot Barril on kaggle.com's
        "House Prices: Advanced Regression Techniques."
        Barril on kaggle.com.

    Attributes:
    -------------
        - gbdt: Gradient boosting decision tree from xgboost package. Created upon fit method invocation.
        - params: Estimator parameters for gradient boosting decision tree
        - nrounds: # of rounds parameter for XGboost estimator
    """
    def __init__(self, seed=0, params=None):
        """
        :param seed: (int) - Random seed
        :param params: (dict) - Estimator parameters
        """
        self.gbdt = None
        self.params = params
        self.params['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)   # pop nrounds from dict w/ 250 as default if not passed

    def train(self, X_train, y_train):
        """
        Method to train a gradient boosting decision tree from the xgboost package, formatted for a generic 'train'
            interface.
        :param X_train: Training features
        :param y_train: Training target/labels
        :return: None
        """
        dtrain = xgb.DMatrix(X_train, y_train)
        self.gbdt = xgb.train(self.params, dtrain, self.nrounds)

    def fit(self, X_train, y_train):
        """
        Method to train a gradient boosting decision tree from the xgboost package, formatted for a
            sklearn-like interface
        :param X_train: Training features
        :param y_train: Training target/labels
        :return: None
        """
        dtrain = xgb.DMatrix(X_train, y_train)
        self.gbdt = xgb.train(self.params, dtrain, self.nrounds)

    def predict(self, X):
        """
        Method calculate estimator predictions
        :param X: Feature set from which to predict targets/labels
        :return: predicted values.
        """
        return self.gbdt.predict(xgb.DMatrix(X))


class SklearnWrapper(object):
    """
    Summary: Wrapper for estimator objects from the XGBoost package. Wraps class to provide access to
        uniform fit/train/predict methods. Derived from original code goes to user Eliot Barril on kaggle.com's
        "House Prices: Advanced Regression Techniques."

    Attributes
    ------------
        - est: Sci-kit learn estimator passed during initialization
        - params: Parameters to pass to the scikit-learn estimator
    """
    def __init__(self, est, seed=0, params=None):
        """
        :param est: (scikit-learn estimator) - Estimator to be wrapped
        :param seed: (int) - Random seed
        :param params: (dict) - Estimator parameters
        """
        self.params = params
        self.params['random_state'] = seed
        self.est = est(**self.params)  # pass dict to

    def train(self, x_train, y_train):
        """
        Method to train estimator package, formatted for a generic 'train' interface.
        :param X_train: Training features
        :param y_train: Training target/labels
        :return: None
        """
        self.est.fit(x_train, y_train)

    def fit(self, x_train, y_train):
        """
        Method to train estimator formatted for a sklearn-like interface
        :param X_train: Training features
        :param y_train: Training target/labels
        :return: None
        """
        self.est.fit(x_train, y_train)

    def predict(self, x):
        """
        Method calculate estimator predictions
        :param X: Feature set from which to predict targets/labels
        :return: predicted values.
        """
        return self.est.predict(x)