# model_selection.py

"""
Module containing functionality for selecting models, including cross validation checks, model scans for common
classifiers/regressors, creation of ensemble models (bagging, majority vote, etc.)
"""

import sys

# Metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
# Classifier import
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Regressor import
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
# Misc ML import
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb

from sklearn.externals import joblib


__all__ = ['clf_scan',
           'reg_scan',
           'reg_scoring',
           'make_majvote',
           'AveragingRegressor',
           'save_model']


# ====== Model Comparisons/scans
def clf_scan(X_train, y_train, X_test=None, y_test=None, cv=5):
    """
    Function to perform k-fold cross validation on some standard classifiers. Note, it may take a long time for
        some of the classifiers to converge on un-scaled data. Use un-scaled data with caution.
    :return: results: Library with classifier names and scores
    :param X_train: Matrix of features from the training set
    :param y_train: Class labels from the training set.
    :param cv: # of folds to use during k-folds cross validation of each model.
    :param X_test: Matrix of features from the testing set
    :param y_test: Class labels from the testing set
    :return: results: Library with classifier names and scores
    """
    rf = RandomForestClassifier(n_estimators=50)
    lr = LogisticRegression()
    mlp = MLPClassifier()
    lda = LinearDiscriminantAnalysis()
    sgd = SGDClassifier()
    ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50)
    gbc = GradientBoostingClassifier()
    svc = SVC(kernel='rbf', probability=True)
    knn = KNeighborsClassifier()
    et = ExtraTreeClassifier()

    clf_names = ['LogisticRegression', 'MLPClassifier', 'LinearDicriminantAnalysis',
                 'SGD Classifier', 'AdaBoostClassifier', 'GradientBoostClassifier', 'SVC(rbf)',
                 'KNearestNeighbors', 'ExtraTreesClassifier', 'RandomForestClassifier']
    clfs = [lr, mlp, lda, sgd, ada, gbc, svc_l, knn, et, rf]

    results = {}
    print('\n====== > Performing cross validation')
    for name, clf in zip(clf_names, clfs):
        print('==> Current estimator:\n%s\n' % clf)
        scores = cross_val_score(clf, X_train, y_train, cv=cv)
        results[name] = scores
    # for name, scores in results.items():
    for name in clf_names:
        print("%25s :: Accuracy: %0.3f%% (+/0 %0.3f%%)" % (name, 100 * results[name].mean(),
                                                           100 * results[name].std() * 2))

    if (X_test is not None) and (y_test is not None):
        test_results = {}
        cohen_kappa_results = {}
        print('=========================================================')
        print('Calculating analagous model fits on training data.')

        for name, clf in zip(clf_names, clfs):
            print('Processing %30s' % name)
            try:
                clf.fit(X_train, y_train)
                test_score = clf.score(X_test, y_test)
                test_results[name] = test_score

                y_pred = clf.predict(X_test)
                kappa = cohen_kappa_score(y_test, y_pred)
                cohen_kappa_results[name] = kappa
            except Exception as e:
                print('Error encountered calculating score on test data for %s. It may not have a built-in'
                      '.score attribute!' % name)
                print('Exception: ', e)
        print('\nNote, Scores on testing data should not necessarily be taken at face value. '
              'In the case of classification problems, classification reports and confusion matrices should '
              'be explored before making a final choice of model.')
        print('=========================================================')

        for name in clf_names:
            print("%25s :: Accuracy:        %0.3f%%\n"
                  "%25s :: Cohen's Kappa:   %0.3f" % (name, 100 * test_results[name],
                                                      " ", cohen_kappa_results[name]))

    return results


def reg_scan(X_train, y_train, cv=5, X_test=None, y_test=None, extra_regressors=None):
    """
    Function to perform k-fold cross validation on some standard regressors. Note, it may take a long time for
        some of the models (e.g., SVM, MLP) to converge on un-scaled data. Use un-scaled data with caution.
    :return: results: Library with regressor names and scores
    :param X_train: (pandas df) Training feature set
    :param y_train: (pandas series/df) - Training target variable set
    :param cv: (int) - # of folds to use during k-folds cross validation of each model.
    :param X_test: Matrix of features from the testing set
    :param y_test: Target variable from the testing set
    :param extra_regressors: (dict, optional) - Key:Value pairs of regressor_name:estimator object. Extra regressors
            to include in the scan aside from those already included.
    :return: results: (dict) Regressor names and scores
    """
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=100)
    etr = ExtraTreesRegressor(n_estimators=100)
    gbr = GradientBoostingRegressor(n_estimators=100)
    ada = AdaBoostRegressor(n_estimators=100)
    svr = SVR(kernel='rbf')

    reg_names = ['LinearRegression', 'Random Forest', 'Extra Trees',
                 'Gradient Boost', 'AdaBoost', 'SVR(rbf)']
    regs = [lr, rf, etr, gbr, ada, svr]

    # Add extra regressors if supplied
    if extra_regressors is not None:
        if not isinstance(extra_regressors, (dict, OrderedDict)):
            raise TypeError("Parameter 'extra_regressors' must be of type dict or collections.OrderedDict")
        for k, v in extra_regressors.items():
            reg_names.append(k)
            regs.append(v)

    results = {}
    print('\n====== > Performing cross validation')
    for name, reg in zip(reg_names, regs):
        print('==> Current estimator:\n%s\n' % reg)
        scores = cross_val_score(reg, X_train, y_train, cv=cv)
        results[name] = scores

    for name in reg_names:
        print("%25s :: CV Score: %0.3f%% (+/0 %0.3f)" % (name, results[name].mean(),
                                                           results[name].std()))
    return results


# ====== Scoring-specific functions
def reg_scoring(y_test, y_pred):
    # Moved to pymlkit
    """
    Function to calculate basic scoring metrics for regression predictions, including r^2, MAE, MdAE and MSE
    :param y_test: (np.array/pandas series) - Actual target variable values
    :param y_pred: (np.array/pandas series) - Predicted target variable values
    :return: mscores: (dict) - dictionary of scores with key:value pairs of scorer_names:scores.
    """
    scorers = [r2_score, median_absolute_error, mean_squared_error, mean_absolute_error]
    scorer_names = ['r^2 Score', 'Median Absolute Error', 'Mean Squared Error', 'Mean Absolute Error']
    mscores = {}

    for sc, sc_name in zip(scorers, scorer_names):
        tmp_score = sc(y_true=y_test, y_pred=y_pred)
        mscores.setdefault(sc_name, tmp_score)
        print('%25s: %0.3f' % (sc_name, tmp_score))

    return mscores


# ====== Ensemble Classifier/Regressor Model Creation
def make_majvote(estimators=None, estimator_names=None, vote_meth='hard'):
    """
    Function to combine multiple majority vote classifiers into a majority vote classifier. The process is not
    complicated, but this function was created to streamline the creation process
    :param estimators: (objects) List of classifiers to be combined in the Voting classifier
    :param estimator_names (strings) List of classifier names as plain text
    :param vote_meth: (string) The voting method to be used, as passed into the VotingClassifier parameter 'voting.'
            Options include: {'hard', 'soft'}
    :return: vc: Majority vote classifier comprised of the classifiers passed as a list in parameter estimators
        with the corresponding names in estimator names
    """
    if len(estimators) != len(estimator_names):
        print('Error in parameters for function "make_mvote." List "estimators" and "estimator_names" must be '
              'of equal length')
        sys.exit()
    else:
        est_list = []
        for est, est_name in zip(estimators, estimator_names):
            est_list.append((est_name, est))
        vc = VotingClassifier(estimators=est_list,
                              voting=vote_meth)

    return vc


class AveragingRegressor(BaseEstimator, RegressorMixin):
    # TODO: Modify how parameters are used. Put them all into a dict. Also change X_train, y_train to just X,y
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


# ====== Wrappers for ensuring methods consistency between ML packages
class XgbWrapper(object):
    """
    Summary: Wrapper for estimator objects from the XGBoost package. Wraps class to provide access to
        uniform fit/train/predict methods. Credit for original code goes to user Eliot
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
        uniform fit/train/predict methods. Credit for original code goes to user Eliot Barril and posted to
        kaggle.com for "House Prices: Advanced Regression Techniques."

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


# ====== Misc/Other
def save_model(model, fname='model'):
    """
    Function to save model to disk as *.pkl file
    :param model: model to be saved
    :param fname: Name for file to which model will be saved
    :return: Dumps model's *.pkl file to current directory
    """
    print('Saving %s \n\n...to file %s.\n' % (model, fname+'.pkl'))
    joblib.dump(model, fname+'.pkl')

    return None
