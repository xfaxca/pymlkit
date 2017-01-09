# model_eval.py

"""
Module containing functions for the evaluation of trained models, including cross validation, classification reports,
ROC AUC curve plotting, confusion matrices and statistical analysis relative to known model outputs.
"""

# Basic Module import
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===================================MODEL METRICS FUNCTIONS=============================================== #
def clf_assessment(clf, X_train, X_test, y_train, y_test, clf_name='Model', oob=False, k=5):
    """
    Function to calculate classification statistics and generate for several metrics, including the confusion matrix,
    classification report, and ROC.
    :param clf: (Obj) The model that will be assessed.
    :param clf_name: (string) Name/type of the classifier being passed
    :param X_train: Training features
    :param X_test: Testing features
    :param y_train: Training class labels
    :param y_test: Testing class labels
    :param oob: (bool) Whether or not to calculate the OOB score. Only set to True if the classifier being passed
        has OOB functionality.
    :param k: (int) k for k-folds cross-validation scoring.
    :return: None
    """
    from sklearn.metrics import log_loss, classification_report, confusion_matrix, cohen_kappa_score
    from sklearn.model_selection import cross_val_score

    # Calculate cross validation score
    scores = cross_val_score(clf, X_train, y_train, cv=k)
    print('CV Score of %s : %0.2f%% (+/- %0.2f%%)' % (clf_name, 100 * scores.mean(),
                                                                    100 * scores.std()))

    clf.fit(X_train, y_train)
    print('%s accuracy score on testing data:' % clf_name, clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    y_predict_proba = clf.predict_proba(X_test)

    print('=========================================================================')
    print('Classification report for %s:\n' % clf_name, classification_report(y_true=y_test, y_pred=y_pred))
    print('=========================================================================')
    sns.heatmap(data=confusion_matrix(y_test, y_pred), annot=True)
    plt.title('%s Confusion Matrix' % clf_name)
    plt.show()

    print('=========================================================================')
    if oob:
        print('OOB Score:           %0.2f' % clf.oob_score_)
    print('Log-Loss:            %0.2f' % log_loss(y_true=y_test, y_pred=y_predict_proba))
    print('Cohen Kappa Score:   %0.2f' % cohen_kappa_score(y_test, y_pred))
    print('=========================================================================')

    return None


# =====================================STATISTICS FUNCTIONS================================================ #
def logloss(actual, predicted_prob):
    # TODO: needs further testing compared to sklearn's implementation
    """
    Function to calculate log loss for a set of binary classifications compared to the
        actual class labels.
    :param actual: actual, ground-truth class labels
    :param predicted_prob: Predicted probabilities of being a specific class.
    :return:
    """
    import scipy as sp
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(actual*sp.log(predicted_prob) + sp.subtract(1, actual)*sp.log(sp.subtract(1, predicted_prob)))
    ll = ll * -1.0/len(actual)


# ===================================VISUALIZATION FUNCTIONS=============================================== #
def confmat_plot(y_pred, y_actual):
    # TODO: Test further
    """
    Function to plot the confusion matrix for classification results.
    :param y_pred: Predicted class labels
    :param y_actual: Actual Class labels
    :return:
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    confmat = confusion_matrix(y_true=y_actual, y_pred=y_pred)
    confplot = sns.heatmap(data=confmat, annot=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
