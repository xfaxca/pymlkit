"""
Module containing functionality for selecting models, including cross validation checks, model scans for common
classifiers/regressors, creation of ensemble models (bagging, majority vote, etc.)
"""
import sys
from collections import OrderedDict

# Metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
# Classifiers
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
# Regressor import
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
# Misc ML import

__all__ = [
    'clf_scan',
    'reg_scan',
    'reg_scoring',
    'make_majvote'
]


# ====== Model Comparisons/scans
def clf_scan(xtrain, ytrain, xtest=None, ytest=None, cv=5):
    """
    Function to perform k-fold cross validation on some standard classifiers. Note, it may take a long time for
        some of the classifiers to converge on un-scaled data. Use un-scaled data with caution.
    :return: results: Library with classifier names and scores
    :param xtrain: Matrix of features from the training set
    :param ytrain: Class labels from the training set.
    :param cv: # of folds to use during k-folds cross validation of each model.
    :param xtest: Matrix of features from the testing set
    :param ytest: Class labels from the testing set
    :return: results: Library with classifier names and scores
    """
    clfs = {
        'LogisticRegression': LogisticRegression(),
        'MLPClassifier': MLPClassifier(),
        'LinearDicriminantAnalysis':  LinearDiscriminantAnalysis(),
        'SGD Classifier': SGDClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50),
        'GradientBoostClassifier': GradientBoostingClassifier(),
        'SVC(rbf)': SVC(kernel='rbf', probability=True),
        'KNearestNeighbors': KNeighborsClassifier(),
        'ExtraTreesClassifier': ExtraTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=50)
    }

    results = {}
    print('\n====== > Evaluation cross validation scores')
    for name, clf in clfs.items():
        print('==> Current estimator:\n%s\n' % clf)
        scores = cross_val_score(clf, xtrain, ytrain, cv=cv)
        results[name] = scores
    # for name, scores in results.items():
    for name in clfs.keys():
        print("%25s :: Accuracy: %0.3f%% (+/0 %0.3f%%)" % (name, 100 * results[name].mean(),
                                                           100 * results[name].std() * 2))

    if (xtest is not None) and (ytest is not None):
        test_results = {}
        cohen_kappa_results = {}
        print('=========================================================')
        print('Performing model fits on training/testing data.')
        for name, clf in clfs.items():
            print('Processing %30s' % name)
            try:
                clf.fit(xtrain, ytrain)
                test_score = clf.score(xtest, ytest)
                test_results[name] = test_score

                y_pred = clf.predict(xtest)
                kappa = cohen_kappa_score(ytest, y_pred)
                cohen_kappa_results[name] = kappa
            except Exception as e:
                print('Error encountered calculating score on test data for %s. It may not have a built-in'
                      '.score attribute!' % name)
                print('Exception: ', e)
        for name in clfs.keys():
            print("%25s :: Accuracy:        %0.3f%%\n"
                  "%25s :: Cohen's Kappa:   %0.3f" % (name, 100 * test_results[name],
                                                      " ", cohen_kappa_results[name]))
    return results


def reg_scan(xtrain, ytrain, cv=5, extra_regressors=None):
    """
    Function to perform k-fold cross validation on some standard regressors. Note, it may take a long time for
        some of the models (e.g., SVM, MLP) to converge on un-scaled data. Use un-scaled data with caution.
    :return: results: Library with regressor names and scores
    :param xtrain: (pandas df) Training feature set
    :param ytrain: (pandas series/df) - Training target variable set
    :param cv: (int) - # of folds to use during k-folds cross validation of each model.
    :param extra_regressors: (dict, optional) - Key:Value pairs of regressor_name:estimator object. Extra regressors
            to include in the scan aside from those already included.
    :return: results: (dict) Regressor names and scores
    """
    regs = {
        'LinearRegression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100),
        'Gradient Boost': GradientBoostingRegressor(n_estimators=100),
        'AdaBoost': AdaBoostRegressor(n_estimators=100),
        'SVR(rbf)': SVR(kernel='rbf')
    }

    # Add extra regressors if supplied
    if extra_regressors is not None:
        if not isinstance(extra_regressors, (dict, OrderedDict)):
            raise TypeError("Parameter 'extra_regressors' must be of type dict or collections.OrderedDict")
        for k, v in extra_regressors.items():
            regs[k] = v

    results = {}
    print('\n====== > Performing cross validation')
    for name, reg in regs.items():
        print('==> Current estimator:\n%s\n' % reg)
        scores = cross_val_score(reg, xtrain, ytrain, cv=cv)
        results[name] = scores

    for name in regs.keys():
        print("%25s :: CV Score: %0.3f%% (+/0 %0.3f)" % (name, results[name].mean(),
                                                         results[name].std()))
    return results


# ====== Scoring-specific functions
def reg_scoring(y_test, y_pred, scorers=None, verbose=0):
    # Moved to pymlkit
    """
    Function to calculate basic scoring metrics for regression predictions, including r^2, MAE, MdAE and MSE
    :param y_test: (np.array/pandas series) - Actual target variable values
    :param y_pred: (np.array/pandas series) - Predicted target variable values
    :param scorers: (optional, dict) - Mapping of scorer name:scoring fucntion. E.g., {'r2': r2_score]
    :return: scores: (dict) - dictionary of scores with key:value pairs of scorer_names:scores.
    """
    if scorers and not isinstance(scorers, dict):
        raise TypeError("'scorers' must be of type dict.")
    if scorers:
        scoring_fns, score_names = list(scorers.keys()), list(scorers.values())
    else:
        scoring_fns = [r2_score, median_absolute_error, mean_squared_error, mean_absolute_error]
        score_names = ['r^2 Score', 'Median Absolute Error', 'Mean Squared Error', 'Mean Absolute Error']

    scores = {}
    for sc, sc_name in zip(scoring_fns, score_names):
        scores[sc_name] = sc(y_true=y_test, y_pred=y_pred)
        if verbose:
            print('%25s: %0.3f' % (sc_name, scores[sc_name]))
    return scores


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
