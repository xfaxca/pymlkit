"""
Module for general input/output and other utility functions.
"""

__all__ = [
    'keys_exist',
    'nested_get'
]


def keys_exist(d, *keys):
    """
    Check if *keys (nested) exists in `element` (dict).
    :param d: (dict) - Dictionary in which to check for key sequences.
    :param keys: (any valid dict key type) - A sequences of nested keys to check for.
    :return:
    """
    if type(d) is not dict:
        raise AttributeError('keys_exists() expects dict as first argument.')
    if len(keys) == 0:
        raise AttributeError('keys_exists() expects at least two arguments, one given.')

    _element = d
    for key in keys:
        try:
            _element = _element[key]
        except (KeyError, TypeError):
            return False
    return True


def nested_get(d, *keys, default=None):
    """
    Retrieves the value of a key that is nested several levels deep in a dictionary. If the key sequence in *keys
        does not exist, then None is return like a normal dict.get().
    :param d: (dict) - The dictionary
    :param keys: A sequences of nested dictionary keys, where the value of the last in the sequences is the target
        of the get.
    :return:
    """
    if keys_exist(d, *keys):
        _val = d[keys[0]]
        for k in keys[1:]:
            _val = _val[k]
        return _val
    return default
