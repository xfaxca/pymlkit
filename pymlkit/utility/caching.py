"""
Module for caching/memoization utility functions
"""

__all__ = [
    'memoize'
]


def memoize(func):
    """
    Memoizes a function by adding an arg cache for store previous results of function calls.
    :param func: (function)
    :return:
    """
    cache = dict()

    def memfunc(*args):
        if args in cache:
            return cache[args]
        cache[args] = func(*args)
        return cache[args]

    return memfunc
