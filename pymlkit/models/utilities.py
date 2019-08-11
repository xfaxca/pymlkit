"""
Module for misc utility functions for model training, evaluation,
"""
from sklearn.externals import joblib

__all__ = [
    'save_model'
]


def save_model(model, fname='model'):
    """
    Function to save model to disk as *.pkl file
    :param model: model to be saved
    :param fname: Name for file to which model will be saved
    :return: Dumps model's *.pkl file to current directory
    """
    print('Saving %s \n\n...to file %s.\n' % (model, fname+'.pkl'))
    joblib.dump(model, fname+'.pkl')
