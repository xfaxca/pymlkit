"""
Module for some simple statistical metrics.
"""
import numpy as np
from sklearn.metrics import r2_score, median_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error

__all__ = [
    'median_abs_pc_err',
    'mean_abs_pc_err',
    'abs_pc_err',
    'r2_score',
    'mean_absolute_error',
    'mean_squared_error',
    'mean_squared_log_error',
    'median_absolute_error'
]


def median_abs_pc_err(y, y_pred):
    """
    Calculates the median abosolute percentage error between two
        numpy arrays.
    :param y: (np.ndarray) - True values
    :param y_pred: (np.ndarray) - Predicted values
    """
    return np.median(abs_pc_err(y, y_pred))


def mean_abs_pc_err(y, y_pred):
    """
    Calculates the mean abosolute percentage error between two
        numpy arrays.
    :param y: (np.ndarray)
    :param y_pred: (np.ndarray)
    """
    return np.mean(abs_pc_err(y, y_pred))


def abs_pc_err(y, y_pred):
    """
    Calculates absolute percentage error
    """
    return (abs(y - y_pred) / y) * 100.0
