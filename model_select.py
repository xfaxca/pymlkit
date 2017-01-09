# model_selection.py

"""
Module containing functions for selection models, including cross validation checks, model scans for common
classifiers/regressors, creation of ensemble models (bagging, majority vote, etc.)

"""


import sys

# Feature selection algorithms
# TODO: look into 'select from model' as well as other feature selection methods. Test these out on the spark
#  Other univariate feature selection methods:
#       1. SelectKBest
#       2. SelectPercentile
#       2.5 SelectFromModel
#       3. SelectFpr (false positive rate)
#       4. SelectFdr (false discover rate)
#       5. SelectFwe (family wise error)
#       6. GenericUnivariateSelect


def clf_scan_full(X_train, y_train, cv=10, X_test=None, y_test=None):
    """
    Function to perform k-fold cross validation on some standard classifiers. Note, it may take a long time for
        some of the classifiers to converge on un-scaled data. Use un-scaled data with caution.
    :return: results: Library with classifier names and scores
    :param X_train: Matrix of features from the training set
    :param y_train: Class labels from the training set.
    :param cv: # of folds to use during k-folds cross validation of each model.
    :param X_test: Matrix of features from the testing set
    :param y_test: Class labels from the testing set
    :return:
    """
    # Metric import
    from sklearn.metrics import cohen_kappa_score
    from sklearn.model_selection import cross_val_score
    # classifier import
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    # Instantiation
    rf = RandomForestClassifier(n_estimators=50)
    lr = LogisticRegression()
    mlp = MLPClassifier()
    lda = LinearDiscriminantAnalysis()
    sgd = SGDClassifier()
    ada = AdaBoostClassifier()
    gbc = GradientBoostingClassifier()
    svc_l = SVC(kernel='linear')
    knn = KNeighborsClassifier()
    #
    clf_names = ['RandomForest', 'LogisticRegression', 'MLPClassifier', 'LinearDicriminantAnalysis',
                 'SGD Classifier', 'AdaBoostClassifier', 'GradientBoostClassifier', 'Linear SVC', 'KNearestNeighbors']
    clfs = [rf, lr, mlp, lda, sgd, ada, gbc, svc_l, knn]

    # Calculate/display cross-validation scores
    results = {}
    for name, clf in zip(clf_names, clfs):
        scores = cross_val_score(clf, X_train, y_train, cv=cv)
        results[name] = scores
    for name, scores in results.items():
        print("%25s :: Accuracy: %0.3f%% (+/0 %0.3f%%)" % (name, 100 * scores.mean(),
                                                           100 * scores.std() * 2))
    # Calculate/display scores on testing data.
    test_results = {}
    cohen_kappa_results = {}
    if (X_test is not None) and (y_test is not None):
        print('=========================================================')
        print('Calculating analagous model fits on training data.')
        for name, clf in zip(clf_names, clfs):
            print('Processing %20s' % name)
            try:
                clf.fit(X_train, y_train)
                # Get test scores (accuracy)
                test_score = clf.score(X_test, y_test)
                test_results[name] = test_score
                # Get Cohen Kappa Coeficient
                y_pred = clf.predict(X_test)
                kappa = cohen_kappa_score(y_test, y_pred)
                cohen_kappa_results[name] = kappa
            except Exception as e:
                print('Error encountered calculating score on test data for %s. It may not have a built-in'
                      '.score attribute!' % name)
                print('Exception: ', e)
        print('\nNote, Scores on testing data should not necessarily be taken at face value. \n'
              'In the case of classification problems, classification reports and confusion matrices should\n'
              'be explored before making a final choice of model.')
        print('=========================================================')
    # Print out accuracies on the test results and the cohen kappa coefficients
    # for name, accuracy, kappa in zip(test_results.keys(), test_results.values(), cohen_kappa_results.values()):
    for name in clf_names:
        print("%25s :: Accuracy:        %0.3f%%\n"
              "%25s :: Cohen's Kappa:   %0.3f" % (name, 100 * test_results[name],
                                                  " ", cohen_kappa_results[name]))
    return results
# TODO: Add auc score to full scan function


def clf_scan_lite(X_train, y_train, cv=10, X_test=None, y_test=None):
    """
    Function to scan several popular classifiers, with those that are typically slow to converge on un-scaled data
        not used. K-fold cross validation is used for comparisons.
    :return: results: Library with classifier names and scores
    :param X_train: Matrix of features from the training set
    :param y_train: Class labels from the training set.
    :param cv: # of folds to use during k-folds cross validation of each model.
    :param X_test: Matrix of features from the testing set
    :param y_test: Class labels from the testing set
    :return:
    """
    # Metric import
    from sklearn.metrics import accuracy_score, log_loss, cohen_kappa_score
    from sklearn.model_selection import cross_val_score, cross_val_predict
    # classifier import
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    # instantiation
    rf = RandomForestClassifier(n_estimators=50)
    lr = LogisticRegression()
    mlp = MLPClassifier()
    lda = LinearDiscriminantAnalysis()
    sgd = SGDClassifier()
    ada = AdaBoostClassifier()
    gbc = GradientBoostingClassifier()
    #
    clf_names = ['RandomForest', 'LogisticRegression', 'MLPClassifier', 'LinearDicriminantAnalysis',
                 'SGD Classifier', 'AdaBoostClassifier', 'GradientBoostClassifier']
    clfs = [rf, lr, mlp, lda, sgd, ada, gbc]
    # Calculate/display cross-validation scores
    results = {}
    for name, clf in zip(clf_names, clfs):
        scores = cross_val_score(clf, X_train, y_train, cv=cv)
        results[name] = scores
    for name, scores in results.items():
        print("%25s :: Accuracy: %0.3f%% (+/0 %0.3f%%)" % (name, 100 * scores.mean(),
                                                           100 * scores.std() * 2))
    # Calculate/display scores on testing data.
    test_acc_results = {}
    cohen_kappa_results = {}
    roc_auc_results = {}
    if (X_test is not None) and (y_test is not None):
        print('=========================================================')
        print('Calculating analagous model fits on training data.')
        for name, clf in zip(clf_names, clfs):
            print('Processing %20s' % name)
            try:
                clf.fit(X_train, y_train)
                # Get test scores (accuracy)
                test_score = clf.score(X_test, y_test)
                test_acc_results[name] = test_score
                # Get Cohen Kappa Coeficient
                y_pred = clf.predict(X_test)
                kappa = cohen_kappa_score(y_test, y_pred)
                cohen_kappa_results[name] = kappa
                try:
                    y_proba = clf.predict_proba(X_test)
                    roc_auc_score = roc_auc_score(y_true=y_test, y_score=y_proba)
                    roc_auc_results[name] = roc_auc_score
                except Exception as e:
                    print('Exception for predicting probabilities for %s. No method.' % name)
                    roc_auc_results[name] = 'N/A'
            except Exception as e:
                print('Error encountered calculating score on test data for %s. It may not have a built-in'
                      '.score attribute!' % name)
                print('Exception: ', e)
        print('\nNote, Scores on testing data should not necessarily be taken at face value. \n'
              'In the case of classification problems, classification reports and confusion matrices should\n'
              'be explored before making a final choice of model.')
        print('=========================================================')
    # Print out accuracies on the test accuracy results, cohen kappa coefficients and roc auc scores
    for name in clf_names:
        print("%25s :: Accuracy:        %0.3f%%\n"
              "%25s :: Cohen's Kappa:   %0.3f" % (name, 100 * test_acc_results[name],
                                                  " ", cohen_kappa_results[name]))
    return results
# TODO: make a function to scan a custom list of estimators, taking name and estimator lists


# Feature selection algorithms
def rfe_cv(est, X, y, cv_method='skf', score_metric=0, transform_x=False, est_name="Model"):
    """
    Function to perform feature selection. Note, this was adapted from an example on the Sci-kit learn website.
    :param est: The estimator to use with RFE
    :param X: Features from dataset with which to perform cross validation
    :param y: Class labels from dataset with which to perform cross validation
    :param cv_method: ({'skf', 'sss'}) The cross-validation method to use. 'skf' = StratifiedKFolds,
        'sss' = StratifiedShuffleSplit
    :param score_meth: (int) Option for the scoring metric to be used for RFECV. Options include: {0: accuracy_score,
        1: log_loss} - see function choose_scoring_classif for more details and options.
    :param transform_x: (bool) Option to return a transformed dataset with the optimal number of
        features as determined by RFECV
    :param est_name: (string) Text name of the estimator being passed for rfe/cross validation
    :return:
    """
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    import matplotlib.pyplot as plt
    from sklearn.metrics import make_scorer

    # Assign appropriate cross-validation and scoring methods according to user input
    if cv_method == 'skf':
        cvm = StratifiedKFold(n_splits=3, shuffle=False)    # default 3 folds, not shuffled
    elif cv_method == 'sss':
        cvm = StratifiedShuffleSplit(n_splits=10, test_size=0.2)    # default params
    else:
        print('ERROR while performing RFE with cross-validation. Invalid choice for parameter cv_meth!')
        return None
    scoring = choose_scoring_classif(choice=score_metric)

    rfe_cv = RFECV(estimator=est, step=1, cv=cvm, scoring=scoring)   # step = # of features to remove each iteration
    rfe_cv.fit(X, y)

    print('The optimal number of features using %s: \n%d' % (est, rfe_cv.n_features_))

    # plot # of features vs cross-validation score
    plt.figure(figsize=(12, 7))
    plt.xlabel('Number of features selected')
    plt.ylabel('Cross validation score (# of correct classifications)')
    plt.plot(range(1, len(rfe_cv.grid_scores_) + 1), rfe_cv.grid_scores_)
    plt.title("RFE Results for %s" % est_name, fontsize=16)
    plt.show()
    # Transform features and return the transformed data
    if transform_x:
        # TODO: need to test this functionality
        return rfe_cv.transform(X), rfe_cv.get_support()
    else:
        return None


# Ensemble model creation (e.g., bagging, majority vote, stacking, etc).
def make_mvote(estimators=None, estimator_names=None, vote_meth='hard'):
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
    from sklearn.ensemble import VotingClassifier

    if len(estimators) != len(estimator_names):
        print('Error in parameters for function "make_mvote." List "estimators" and "estimator_names" must be '
              'of equal length')
        sys.exit()
    else:
        est_list = []
        for est, est_name in zip(estimators, estimator_names):
            est_list.append((est_name, est))
        print('test out for est_list')
        vc = VotingClassifier(estimators=est_list,
                              voting=vote_meth)
    return vc


# Utility functions (for use by other functions in this module)


# GridSearch/Parameter optimization algorithms
def gs_rf(X_train, y_train):
    """
    Grid search for Random Forest Classifier
    :param X_train: Training features
    :param y_train: Training class labels
    :return:
    """
    from sklearn.grid_search import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier()

    param_grid = [{'n_estimators': [10, 100],
                   'criterion': ['gini', 'entropy'],
                   'min_samples_leaf': [1, 10, 50],
                   'min_samples_split': [2, 5, 10],
                   'max_features': ['auto', 'sqrt']
                   }]
    gs = GridSearchCV(estimator=rf,
                      param_grid=param_grid,
                      scoring=None,
                      cv=5,
                      n_jobs=1)
    gs.fit(X_train, y_train)

    best_est = gs.best_estimator_
    best_params = gs.best_params_
    best_score = gs.best_score_
    print('Best score from Random Forest GridSearch is:', best_score)
    print('Best parameters from Random Forest GridSearch is:', best_params)
    return best_est, best_params, best_score


def gs_mlp(X_train, y_train):
    # TODO: Needs further testing. Some of the parameter combinations may not work well.
    """
    Grid search for Multi-Layer Perceptron Classifier. Warning: may take quite a long time.
    :param X_train: Training features
    :param y_train: Training class labels
    :return:
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier()

    param_grid = [{'hidden_layer_sizes': [(1000, ), (100, 100), (10, 10, 10, 10)],
                   'activation':['identity', 'logistic', 'tanh', 'relu'],
                   'learning_rate': ['constant', 'invscaling', 'adaptive'],
                   'solver': ['lbfgs', 'sgd', 'adam'],
                   'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
                   }]
    gs = GridSearchCV(estimator=mlp,
                      param_grid=param_grid,
                      scoring=None,
                      cv=5,
                      n_jobs=1)
    gs.fit(X_train, y_train)

    best_est = gs.best_estimator_
    best_params = gs.best_params_
    best_score = gs.best_score_
    print('Best score from MLP GridSearch is:', best_score)
    print('Best parameters from MLP GridSearch is:', best_params)
    return best_est, best_params, best_score


def gs_gbc(X_train, y_train):
    """
    Grid search for GradientBoostingClassifier
    :param X_train: Training features
    :param y_train: Training class labels
    :return:
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier

    gbc = GradientBoostingClassifier()

    param_grid = [{'loss': ['deviance', 'exponential'],
                   'learning_rate': [0.01, 0.1, 1.0],
                   'n_estimators': [100],
                   'max_depth': [3, 5, 10],
                   'criterion': ['friedman_mse', 'mae', 'mse'],
                   'min_samples_leaf': [1, 10]}]
    gs = GridSearchCV(estimator=gbc,
                      param_grid=param_grid,
                      scoring=None,
                      cv=5,
                      n_jobs=1)
    gs.fit(X_train, y_train)

    best_est = gs.best_estimator_
    best_params = gs.best_params_
    best_score = gs.best_score_
    print('Best score from GBC GridSearch is:', best_score)
    print('Best parameters from GBC GridSearch is:', best_params)
    return best_est, best_params, best_score


def gs_lda(X_train, y_train):
    """
    Grid search for Linear Discriminant Analysis
    :param X_train: Training features
    :param y_train: Training class labels
    :return: Best estimator, parameters and accuracy score
    """
    from sklearn.grid_search import GridSearchCV
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis()

    param_grid = [{'solver': ['svd', 'lsqr', 'eigen'],
                   'n_components': [None, 10, 20],
                   'tol': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]}]
    gs = GridSearchCV(estimator=lda,
                      param_grid=param_grid,
                      scoring=None,
                      cv=5,
                      n_jobs=1)
    gs.fit(X_train, y_train)

    best_est = gs.best_estimator_
    best_params = gs.best_params_
    best_score = gs.best_score_
    print('Best score from Linear Discriminant Analysis GridSearch is:', best_score)
    print('Best parameters from Linear Discriminant Analysis GridSearch is:', best_params)
    return best_est, best_params, best_score


def gs_sgd(X_train, y_train):
    """
    Grid search for Logistic Regression
    :param X_train:
    :param y_train:
    :return:
    """
    from sklearn.grid_search import GridSearchCV
    from sklearn.linear_model import SGDClassifier

    sgdr = SGDClassifier()

    # param_grid = [{'loss': ['hinge', 'log', 'modified_huber'],
    #                'alpha': [0.01, 0.001, 0.0001, 0.00001],
    #                'epsilon': [0.1, 0.01],
    #                'eta0': [0.1, 0.01, 0.001]}]
    param_grid = [{'loss': ['hinge'],
                   'alpha': [0.01, 0.001, 0.0001, 0.00001],
                   'epsilon': [0.1, 0.01],
                   'eta0': [0.1, 0.01, 0.001]}]
    gs = GridSearchCV(estimator=sgdr,
                      param_grid=param_grid,
                      scoring=None,
                      cv=3,
                      n_jobs=1)
    gs.fit(X_train, y_train)

    best_est = gs.best_estimator_
    best_params = gs.best_params_
    best_score = gs.best_score_
    print('Best score from SGD classifier gridsearch is:', best_score)
    print('Best parameters from SGD classifier gridsearch is:', best_params)
    return best_est, best_params, best_score


def gs_lr(X_train, y_train):
    """
    Grid search for Logistic Regression
    :param X_train:
    :param y_train:
    :return:
    """
    from sklearn.grid_search import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()

    # param_grid = [{'penalty': ['l2'],
    #                'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
    #                'solver': ['liblinear', 'lbfgs']}]
    param_grid = [{'penalty': ['l2'],
                   'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
                   'fit_intercept': [True, False],
                   'intercept_scaling': [0.001,.01, .1, 1.0],
                   'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag'],
                   'max_iter': [100, 200]}]
    gs = GridSearchCV(estimator=lr,
                      param_grid=param_grid,
                      scoring=None,
                      cv=3,
                      n_jobs=1)
    gs.fit(X_train, y_train)

    best_est = gs.best_estimator_
    best_params = gs.best_params_
    best_score = gs.best_score_
    print('Best score from Logistic Regression gridsearch is:', best_score)
    print('Best parameters from Logistic Regression gridsearch is:', best_params)
    return best_est, best_params, best_score


def gs_svc(X_train, y_train):
    """
    Grid search for Logistic Regression
    :param X_train:
    :param y_train:
    :return:
    """
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC

    svc = SVC()

    param_grid = [{'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                   'kernel': ['linear', 'rbf', 'poly'],
                   'degree': [3, 4, 5],
                   'gamma': ['auto'],
                   'probability': [True],
                   'tol': [1e-3],
                   'decision_function_shape': ['ovr']}]
    gs = GridSearchCV(estimator=svc,
                      param_grid=param_grid,
                      scoring=None,
                      cv=3,
                      n_jobs=1)
    gs.fit(X_train, y_train)

    best_est = gs.best_estimator_
    best_params = gs.best_params_
    best_score = gs.best_score_
    print('Best score from Logistic Regression gridsearch is:', best_score)
    print('Best parameters from Logistic Regression gridsearch is:', best_params)
    return best_est, best_params, best_score


# Utilities functions for other functions in this module
def choose_scoring_classif(choice=0):
    # TODO: This is not returning the correct type of metric object
    """
    Function that returns a scorer with an appropriate scoring metric from Sci-kit Learn
    :param choice: (int) Choice of scoring metric. Options include:
    :return:
    """
    from sklearn.metrics import accuracy_score, log_loss, average_precision_score, roc_auc_score
    from sklearn.metrics import f1_score, hinge_loss, precision_score, recall_score
    from sklearn.metrics import make_scorer

    met_lib = {0: accuracy_score, 1: log_loss, 2: average_precision_score, 3: roc_auc_score,
               4: f1_score, 5: hinge_loss, 6: precision_score, 7: recall_score}
    metric = make_scorer(met_lib[choice])
    print('Chosen metric:\n', metric)
    return metric
