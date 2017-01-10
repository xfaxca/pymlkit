# explore.py

"""
Module containing functions for data exploration and visualization.
"""

import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['class_proportions',
           'see_nulls']


# ====== Data Statistics
def class_proportions(y):
    """
    Function to calculate the proportion of classes for a given set of class labels. Returns a dictionary of class
        proportions where the keys are the labels and the values are the percentage of the total number of samples that
        occur for each class label.
    :param y: (int) List of class labels (typically int in classification problems, but can be passed as strings)
    :return:
    """
    if not isinstance(y, list):
        y = list(y)

    counts_dict = {i: y.count(i) for i in y}
    prop_dict = {}
    for key, val in zip(counts_dict.keys(), counts_dict.values()):
        print('Class: %10s | counts: %i (%0.2f%%)' % (key, val, (100 * val / len(y))))
        prop_dict[key] = (100 * val / len(y))
    print('Total number of samples:', len(y))
    return prop_dict


# ====== Visualization
def see_nulls(df):
    """
    Function to visualize columns with null values for features in a pandas DataFrame
    :param df: pandas DataFrame with feature data
    :return:
    """
    plt.figure(figsize=(14, 9))
    sns.heatmap(df.isnull(), cmap='viridis', yticklabels=False, xticklabels=True, cbar=True)
    plt.title("Visualization of Null Values in Data")
    plt.xticks(rotation=30)
    plt.show()

    return None

