# model_selection.py

"""
Module containing functionality for selecting models, including cross validation checks, model scans for common
classifiers/regressors, creation of ensemble models (bagging, majority vote, etc.)
"""

import sys

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.externals import joblib


__all__ = ['clf_scan',
           'make_majvote',
           'save_model']


# ====== Model Comparison
def clf_scan(X_train, y_train, cv=5, X_test=None, y_test=None):
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
    svc_l = SVC(kernel='linear', probability=True)
    knn = KNeighborsClassifier()
    et = ExtraTreeClassifier()

    clf_names = ['LogisticRegression', 'MLPClassifier', 'LinearDicriminantAnalysis',
                 'SGD Classifier', 'AdaBoostClassifier', 'GradientBoostClassifier', 'Linear SVC',
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


# ====== Ensemble Model Creation
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


# ====== Other
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
