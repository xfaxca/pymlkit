"""
Module for tokenization and punctuation treatment.
"""
import re
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

__all__ = [
    'iterate_tokens_from_df',
    'simple_tokenize',
    'remove_punctuation'
]


def iterate_tokens_from_df(df, idcol, textcol, tokenizer_fn=None, **kwargs):
    """
    Returns a generator of id:token pairs, with the tokens being the result of the application of the specified
        tokenizer function to the specified text column in the dataframe.
    :param df: (pd.DataFrame)
    :param idcol: (str) - Name of the column where document/string ids are located.
    :param textcol: (str) - Name of the column where the text to be tokenized is located.
    :param tokenizer_fn: (function) - Function/callable that will be used for tokenization.
    :param strip_punct: (whether or not to strip punctuation out of the text prior to tokenization. NOTE: The tokenizer
        in use ust support the keyword 'strip_punct` for this to have any impact.
    :keywords: Keywords to be passed to the tokenizer being used.
    :return:
    """
    tokenize = tokenizer_fn if tokenizer_fn else simple_tokenize
    for id_tag, text in zip(df[idcol], df[textcol]):
        if pd.isnull(text):
            continue
        tokens = tokenize(text, **kwargs)
        yield id_tag, tokens


def simple_tokenize(text, strip_punct=True, stopwords=None):
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
