# premade.py

"""
Module containing some premade estimators for quick implementation and testing.
"""


# Premade bagging estimators
def bag_of_adac(n_est=10):
    """
    Function create and return a Bagging Classifier comprised of ADABoost classifiers as the base classifier,
        using the base decision estimator for base estimator of Ada
    :param n_est: (int) Number of base ADABoost estimators to use
    :return:
    """
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier

    tree = DecisionTreeClassifier()
    ada = AdaBoostClassifier(base_estimator=tree,
                             n_estimators=100,
                             learning_rate=1.0,
                             algorithm='SAMME.R',
                             random_state=0)
    bag = BaggingClassifier(base_estimator=ada,
                            n_estimators=n_est,
                            max_samples=1.0,
                            max_features=1.0,
                            bootstrap=True,
                            bootstrap_features=False,
                            oob_score=True,
                            warm_start=False,
                            verbose=1)
    return bag


def bag_of_gbc(n_est=10):
    """
    Function create and return a Bagging Classifier comprised of Gradient Boost classifiers as the base classifier
    :param n_est: (int) Number of gbc estimators to bag
    :return:
    """
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier(loss='deviance',
                                     learning_rate=0.1,
                                     n_estimators=100,
                                     subsample=1.0,
                                     criterion='friedman_mse',
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     max_depth=4,
                                     min_impurity_split=1e-07,
                                     init=None,
                                     random_state=None,
                                     warm_start=False,
                                     presort='auto')
    bag = BaggingClassifier(base_estimator=gbc,
                            n_estimators=n_est,
                            max_samples=1.0,
                            max_features=1.0,
                            bootstrap=True,
                            bootstrap_features=False,
                            oob_score=True,
                            warm_start=False,
                            verbose=1)
    return bag


def bag_of_mlpc(n_est=10, hidden_layer_sizes=(100, )):
    """
    Function that creates and returns a bag of Multi-Layer Perceptron classifiers with the specified hidden_layer_sizes
    parameter.
    :param n_est: (int): Number of MLPClassifiers to bag
    :param hidden_layer_sizes: (tuple): Hidden layer dimensions
    :return:
    """
    from sklearn.ensemble import BaggingClassifier
    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
    bag = BaggingClassifier(base_estimator=mlp,
                            n_estimators=n_est,
                            max_samples=1.0,
                            max_features=1.0,
                            bootstrap=True,
                            bootstrap_features=False,
                            oob_score=True,
                            warm_start=False,
                            verbose=1)
    return bag
# TODO - add bags of regressors


# Premade ADAboost estimators with different base_estimators
def adaboost_rfc(n_est=10, n_trees=10):
    """
    Function to return an ADAboost estimator with random forest as the base estimator
    :param n_est: (int) - Number of base estimators (random forests) in the Adaboost estimator.
    :param n_trees (int) - Number of trees in each random forest base estimator
    :return:
    """
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=n_trees,
                                oob_score=True)
    ada = AdaBoostClassifier(base_estimator=rf,
                             n_estimators=n_est,
                             learning_rate=1.0,
                             algorithm='SAMME.R',
                             random_state=0)
    return ada


def adaboost_gbc(n_est=10, n_est_gbc=100):
    # TODO: Needs testing
    """
    Function to return an ADAboost estimator with gradient boost classifier as the base estimator
    :param n_est: (int) - Number of base estimators (random forests) in the Adaboost estimator.
    :param hidden_layer_sizes: (tuple) Specification of the size of the hidden layer in the MLP Classifier.
    :return:
    """
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
    gbc = GradientBoostingClassifier(loss='deviance',
                                     learning_rate=0.1,
                                     n_estimators=n_est_gbc,
                                     criterion='friedman_mse')
    ada = AdaBoostClassifier(base_estimator=rf,
                             n_estimators=n_est,
                             learning_rate=1.0,
                             algorithm='SAMME.R',
                             random_state=0)
    return ada

# TODO: make some premade majority vote (3 and/or 5 estimators of common types).


# ADd functions to create majority vote