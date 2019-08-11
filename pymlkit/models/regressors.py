"""
Module for custom regression model classes.
"""
from sklearn.base import BaseEstimator, RegressorMixin

"""
Rolling todo:
    1. For AvgReg: Modify how parameters are used. Put them all into a dict. Also change X_train, y_train to just X,y
"""


class AveragingRegressor(BaseEstimator, RegressorMixin):
    """
    Summary: A Meta-regressor that averages all predictions of it's consituent regressors. Analogous to
        a majority vote classifer, but for regressoion

    Attributes:
    -------------
        - regs: Base/Constituent regressors from which the average predictions are calculated
        - reg_names: Names of the constituent regressors
        - params: Optionally user-supplied initialization parameters for the
        - base_predictions: Predictions of the constituent classifiers. This attribute is None until the predict method
                is called
        - avg_predictions: Average predictions calculated from the predictions of the constituent regressors.
    """
    def __init__(self, regressors=None, regressor_names=None, init_parameters=None, verbose=0):
        """
        Initialization
        :param regressors: (obj list) - Constituent regressors of AveragingRegressor
        :param regressor_names: (str list) - Names of the constituent regressors
        :param init_parameters: (dict list) - initialization parameters for the corresponding regressors. These
                must be passed as a list of dictionaries s.t. the parameters in each index are the corresponding
                paramters for the regressor at the same index in the 'regressors' parameter. Can provide a partial
                list, containing parameter dictionaries only for the first few regressors.
        """
        self.params = {'regressors':      regressors,
                       'regressor_names': regressor_names,
                       'init_parameters': init_parameters,
                       'verbose': verbose}
        self.regs = regressors
        self.reg_names = regressor_names
        self.reg_params = init_parameters
        self.verbose = verbose
        self.base_predictions = None
        self.avg_predictions = None
        super().__init__()
        super().set_params(**self.params)

        # Return error if no constituent regressors are supplied
        if regressors is None:
            raise TypeError("Parameter 'regressors' should be a list of estimators with base scikit-learn regressor"
                            " methods.")

        # Initialize constituent regressors with custom parameters if they are provided
        if init_parameters is not None:
            for i in range(len(self.reg_params)):
                self.regs[i] = self.regs[i](**self.reg_params[i])

    def fit(self, X_train, y_train=None):
        """
        Method to fit all Regressors
        :param X_train: (pandas df) - Training features
        :param y_train: (pandas series) - Training target variable
        :return: None
        """
        print('=> Fitting AveragingRegressor:')
        for i in range(len(self.regs)):
            if self.verbose > 0:
                print('==> Fitting %s' % self.reg_names[i])
            self.regs[i].fit(X_train, y_train)

    def predict(self, X_test):
        """
        Method to predict target variable values. Final results are the average of all predictions
        :param X_test: (pandas df) - Test features
        :return: self.avg_predictions: (np.array) Average target variable predictions
        """
        predictions = {}
        average_predictions = np.zeros(shape=(len(X_test)), dtype=np.float64)

        if len(self.reg_names) == len(self.regs):
            add_names = True
        for i in range(len(self.regs)):
            y_pred = self.regs[i].predict(X_test)
            average_predictions += y_pred
            name = self.reg_names[i] if add_names else ('Regressor%i' % i)
            predictions.setdefault(name, y_pred)

        average_predictions /= float(len(self.regs))
        predictions.setdefault('Average', average_predictions)
        self.base_predictions = predictions
        self.avg_predictions = average_predictions

        return self.avg_predictions