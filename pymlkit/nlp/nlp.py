# nlp.py

"""
Module containing natural language processing wrapper functions for Sci-kit Learn as well as Regex functions

NOTE: This module should be split up into different categories/topics as it grows.
"""
import csv
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import SnowballStemmer, PorterStemmer, WordNetLemmatizer

__all__ = ['firstlast_word_extractor',
           'StemmedCountVectorizer',
           'StemmedTfidfVectorizer',
           'TextPreprocessor']


# ====== Feature Extractors
def firstlast_word_extractor(doc):
    tokens = doc.split()
    first, last = tokens[0], tokens[-1]
    feats = {}
    feats["first({0})".format(first)] = True
    feats["last({0})".format(last)] = False
    return feats


# ====== Classes
class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):
        english_stemmer = SnowballStemmer('english')
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


# Class to vectorize using tfidf and stemming
class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        english_stemmer = SnowballStemmer('english')
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


class TextPreprocessor:
    """
    Class to load and pre-process a body of text. Functions include removal of unwanted characters, removal of
        single quotes (sans apostrophes), tokenization, stemming, stopword removal and others. See individual method
        documentation for more details.

    Attributes:
        1. text: Text represented as a list of lines and/or sentences, depending on how the text was loaded.
        2. tokenized_text: The tokenized results of the contents of self.text that are created when calling the
                           tokenize() method.
    """
    def __init__(self):
        self.text = None
        self.tokenized_text = None

    def read_textfile(self, path, delimiter=None, as_sentences=False):
        """
        Method to read in a text file into attribute self.text. The file is read in as a list, where each element of
            the list corresponds to the contents of the lines of the text file. Note that if self.text is not currently
            empty, the contents of the new file are added in addition to whatever currently exists there.
        :param path: (str) Path to text file to be loaded into self.text
        :param delimiter: (str): Delimiter to split by while reading file
        :param as_sentences: (bool) Choice to import individual sentences instead of simple lines of text. This
            can be useful if the text is complete sentences, such as a book, an essay, etc.
        """
        new_text = []
        if not as_sentences:
            with open(path, 'r') as f:
                raw_text = f.readlines()
                for line in raw_text:
                    if len(line) > 1:
                        new_text.append(" ".join(line.split(sep=delimiter)))
        else:
            with open(path, 'r') as f:
                new_text = TextBlob(f.read()).raw_sentences

        if self.text:
            self.text = self.text + new_text
        else:
            self.text = new_text
        print('Loaded %i new lines of text from text file: %s' % (len(new_text), path))

    def read_csv(self, path, as_sentences=False):
        """
        Method to read in text data from a csv file. The contents are then stored as a list, where each element of the list
            corresponds to the contents of the lines of the text file. Note that if self.text is not currently empty,
            the contents of the new file are added in addition to whatever currently exists there.
        :param path: (str) Path to text file to be loaded into self.text
        :param as_sentences: (bool) Choice to import individual sentences instead of simple lines of text. This
            can be useful if the text is complete sentences, such as a book, an essay, etc.
        """
        new_text = []
        with open(path, 'r') as f:
            if not as_sentences:
                reader = csv.reader(f)
                for line in reader:
                    line = " ".join(line).strip()
                    if len(line) > 0:
                        new_text.append(line)
            else:
                reader = csv.reader(f)
                f_contents = " ".join(" ".join(line).strip() for line in reader)
                new_text = TextBlob(f_contents).raw_sentences

        if self.text:
            self.text = self.text + new_text
        else:
            self.text = new_text
        print('Loaded %i new lines of text from csv file: %s' % (len(new_text), path))

    def add_text(self, new_text):
        """
        Method to add a text/document (as list) to the TextPreprocessor instance. If the attribute self.text is not
            empty, the new text is appended to the existing body of text.
        :param new_text: (list) A list of string values that represents elements of a text, such as individual
            sentences, lines of text, or some other organization of text.
        """
        assert(isinstance(new_text, list))

        if self.text is None:
            self.text = new_text
            print('Loaded text with %i elements.' % len(new_text))
        else:
            for new_line in new_text:
                self.text.append(new_line)
            print('Added %i new lines of text to TextPreprocessor' % (len(self.text) - len(new_text)))

    def clear_text(self, keep_tokenized=False):
        """
        Method to clear all text from the attribute self.text
        :param keep_tokenized: (bool) Option to keep tokenized text and only remove data from the attribute self.text
        """
        if not keep_tokenized:
            self.tokenized_text = None
        self.text = None

    def words_only(self):
        """
        Method to remove all non-word characters from text. Alternative to the 'remove_characters' method if it is
            known that no non-alphabetic characters are desired.
        """
        words_only = []
        for line in self.text:
            words_only.append(re.sub('[^a-zA-Z]', ' ', line))
        self.text = words_only

    def remove_characters(self, characters, replacements=None):
        """
        Method to remove each character from a list of characters provided by the user. Note, if apostrophes/single
            quotes are removed, they are removed from all places, including from conjunctions. A good use of this method
            is the removal of punctuation.
        :param characters: (str/char list) Strings/characters to be removed from the text
        :param replacements: (str/char list) Strings/characters with which to replace the removed characters. Should be
            in the same order as the characters they will replace in param characters.
        """
        assert(isinstance(characters, list))
        assert((replacements is None) or (len(characters) == len(replacements)))

        if replacements is None:
            replacements = len(characters)*[""]

        map_dict = {}
        for char, replacement in zip(characters, replacements):
            map_dict[char] = replacement
        map = str.maketrans(map_dict)

        for lineno in range(len(self.text)):
            self.text[lineno] = self.text[lineno].translate(map)

    def remove_single_quotes(self):
        """
        Method to remove single quotes/hanging apostrophes, but leave those that are in conjunctions.
        """
        for lineno in range(len(self.text)):
            self.text[lineno] = re.sub(r"""["?,$!]|'(?!(?<! ')[ts])""", "", self.text[lineno])

    def tokenize(self, lowercase=False):
        """
        Method to tokenize the current text and return the tokenized text. Note, the text in it's current state
            is tokenized. Therefore, one should take care to perform any desired preprocessing steps before
            tokenization.
        :param lowercase: Option to lowercase all words upon tokenization. Note, this may improve results of subsequent
            stopword removal.
        :return: self.tokenized_text: Text with tokenized elements.
        """
        assert(isinstance(lowercase, bool))

        new_tokenized_text = []
        for lineno in range(len(self.text)):
            if lowercase:
                line = self.text[lineno].lower()
            else:
                line = self.text[lineno]
            new_tokenized_text.append(word_tokenize(text=line, language='english'))

        self.tokenized_text = new_tokenized_text
        return self.tokenized_text

    def lowercase_tokens(self, exclude_list):
        """
        Method to lowercase all words in text. If tokenization has already taken place
        :param exclude_list: List of words to exclude from the transformation to lower case. Note, these should be
            in the original form they appear. Example, list of cities to remain capitalized: ['Atlanta', 'Boston',
            'London', 'Paris', 'New York']
        """
        for tokens in self.tokenized_text:
            for i in range(len(tokens)):
                if tokens[i] not in exclude_list:
                    tokens[i] = tokens[i].lower()

    def remove_stopwords(self, stopwords_list=None, extra_stopwords=None):
        """
        Method to remove stopwords from a tokenized text.
        :param stopwords_list: (optional) List of stopwords to remove. If not provided, then nltk's english stopwords.
            are used. Note, punctuation should be removed prior to invoking this method unless there is a specific
            reason.
        :param extra_stopwords: (optional) List of words as strings that should be removed in addition to nltk's
            default english stopwords (nltk.text.stopwords.words('english)) in the case that a custom stopwords
            list is not provided via the stopwords_list parameter. extra_stopwords is combined with stopwords_list
            if both are provided.
        """
        assert self.tokenized_text

        if stopwords_list:
            if extra_stopwords:
                stopwords_list = stopwords_list + extra_stopwords
            stop = stopwords_list
        else:
            if extra_stopwords:
                stop = stopwords.words('english') + extra_stopwords
            else:
                stop = stopwords.words('english')

        tokenized_text_stopped = []
        for lineno in range(len(self.tokenized_text)):
            tokenized_text_stopped.append([t for t in self.tokenized_text[lineno] if t not in set(stop)])

        self.tokenized_text = tokenized_text_stopped

    def stem_tokens(self, skip_stopwords=False):
        """
        Method to stem the tokens residing in self.tokenized_text using nltk's SnowballStemmer
        :return: Nothing. Modifies tokenized text
        """
        ss = SnowballStemmer(language='english', ignore_stopwords=skip_stopwords)

        lemmatized_tokens = []
        for tokens in self.tokenized_text:
            lemmatized_tokens.append([ss.stem(t) for t in tokens])

        self.tokenized_text = lemmatized_tokens

    def rejoin_tokens(self):
        """
        Method to rejoin tokens into a single string so that it can then be used by an external vectorizer. A
            vectorization method is not included since plenty of external tools exist for this already. Note, this does
            not modify the contents of self.tokenized_text, but only returns the rejoined text as a list of lines or
            sentences (depending on how they were loaded/read in).
        :return: rejoined_tokens: (list) A list of the lines of text, once tokenized, now rejoined. Note that depending
            how certain punctuation marks are handled (e.g., "'" in "don't" or "isn't"), results may vary with respect
            to how closely the rejoined tokens resemble the original text.
        """
        rejoined_tokens = []
        for tokens in self.tokenized_text:
            rejoined_tokens.append(" ".join(tokens))

        return rejoined_tokens
