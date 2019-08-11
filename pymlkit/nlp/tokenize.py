"""
Module for tokenization and punctuation treatment.
"""
import re
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS


def simple_tokenize(text, strip_punct=False, stopwords=None):
    """
    Tokenization function, with option to strip
    """
    if stopwords is None:
        stopwords = STOPWORDS
    if strip_punct:
        return [remove_punctuation(token) for token in simple_preprocess(text) if token not in stopwords]
    else:
        return [token for token in simple_preprocess(text) if token not in stopwords]


def remove_punctuation(s, repl=''):
    """
    Removes punctuation from a string and replaces it with string specified by `repl` arg.
    :param s: (str) - String to be processed.
    :param repl: (str) - Replacement character/string
    :return:
    """
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]", repl, s))
