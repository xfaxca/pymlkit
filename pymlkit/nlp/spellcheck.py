"""
Spellchecker coming soon

Originally inspired by code at http://norvig.com/spell-correct.html
"""
import re
import string
from collections import Counter
from functools import lru_cache
from pymlkit.utility.caching import memoize


class SpellChecker:
    """
    English language spell-checker.

    TODO list:
        1. Make more efficient for large data by discarding unncessary data after load.
        2. Test with much larger corpus.
        3. Support alternative word tokenizers than just split by \w+
    """
    def __init__(self, text=None, autoparse=False):
        self.text = text
        self.letters_lower = string.ascii_lowercase
        # If text is passed and automatic parsing is turned on, parse the text.
        if autoparse and text is not None:
            self.parse_words(text=self.text, lowercase=True)
            self.count_words(words=self.words)
        else:
            self.words = None
            self.n_words = 0
            self.wordcounts = None
            self.n_unique_words = 0
            self.unique_words = None

    def parse_words(self, text, lowercase=True):
        """
        Parses raw text into word tokens
        :param text: (str) - Text to be parsed
        :param lowercase: (bool) - Option to lowercase the text prior to tokenization.
        :return:
        """
        if lowercase:
            self.words = re.findall(r'\w+', text.lower())
        else:
            self.words = re.findall(r'\w+', text)
        self.unique_words = set(self.words)
        return self.words

    def count_words(self, words):
        """
        Applies collections.Counter
        :param words: (list of str) - Words to be counted
        :return:
        """
        self.wordcounts = Counter(words)
        self.n_words = sum(self.wordcounts.values())
        self.n_unique_words = len(self.wordcounts.keys())
        return self.wordcounts

    def probability(self, word):
        """
        Calculates probability of word as # times word occurs in text / total number of words in text.
        :param word:
        :return:
        """
        return self.wordcounts[word] / self.n_words

    @memoize
    def candidates(self, word):
        """
        Finds all possible spelling corrections for the word
        :param word:
        :return:
        """
        return (self.known([word])) or self.known(self.d1_edits(word)) or self.known(self.d2_edits(word) or [word])

    def known(self, words):
        """
        Gets all possible spelling corrections for the passed word.
        :param words: (str) - The word for which to find possible spelling errors.
        :return:
        """
        return set(w for w in words if w in self.unique_words)

    @lru_cache(maxsize=32)
    def correction(self, word, n=None):
        """
        Returns the most probable spelling correction for a word.
        :param word: (str)
        :param n: (int) - Specify the top n correction suggestions. If None, the top suggestion is returned.
        :return:
        """
        candidates = self.candidates(word)
        if candidates and n is None:
            return max(candidates, key=self.probability)
        elif candidates and n >= 0:
            candidates = sorted(candidates, key=lambda x: 1 / self.probability(x))
            return candidates[0:n]
        return 'No suggestions found'

    def d1_edits(self, word):
        """
        Finds all distance 1 edits for a word.
        :param word:
        :return:
        """
        # TODO: add some caching of these results in somewhere.
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [left + right[1:] for left, right in splits if right]  # one-character deletions
        # Transposes, or swapping of letters
        transposes = [left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1]
        # Replacements with another character
        replaces = [left + center + right[1:] for left, right in splits if right for center in self.letters_lower]
        # Extra characters inserted.
        inserts = [left + center + right for left, right in splits for center in self.letters_lower]
        return set(deletes + transposes + replaces + inserts)

    def d2_edits(self, word):
        """
        Finds all distance 2 edits for a word.
        :param word:
        :return:
        """
        return (edit2 for edit1 in self.d1_edits(word) for edit2 in self.d1_edits(edit1))


def main():
    doc = open('../../data/lotr/concerning_hobbits.txt').read()
    print('doc[0]:', doc[0:100])
    sc = SpellChecker()
    sc.parse_words(doc)
    sc.count_words(sc.words)
    print('words[0:100]:', sc.words[0:100])
    print('wordcounts:', sc.wordcounts)
    print("{} total words found. {} Unique words".format(sc.n_words, sc.n_unique_words))
    print("probability of hobbit: ", str(round(sc.probability('hobbit') * 100, 2)) + '%')
    print('probability of the word "the": ', str(round(sc.probability('the') * 100, 2)) + '%')
    m = string.ascii_lowercase
    print(m)
    print(type(m))

    print('correct hobit:', sc.correction('hobit'))
    print('correct pipeweed:', sc.correction('helo'))
    print('hello in words:', 'hello' in sc.unique_words)
    pass


if __name__ == '__main__':
    main()
