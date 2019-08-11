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
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
# Misc ML import

__all__ = [
    'clf_scan',
    'reg_scan',
    'reg_scoring',
    'make_majvote'
]


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
