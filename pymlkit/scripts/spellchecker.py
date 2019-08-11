from pymlkit.nlp.spellcheck import SpellChecker


def main():
    """
    Testing the spellchecker using a corpus based on a couple of chapters from The Hobbit.
    :return:
    """
    # Docs can be any set of test files to read in to use in the spellchecker
    datadir = '../../data/lotr/'
    docs = [
        'concerning_hobbits.txt',
        'concerning_pipeweed.txt',
        'finding_of_the_ring_onwards.txt',
        'ordering_of_the_shire.txt'
    ]
    doctexts = []
    for doc in docs:
        doctexts.append(open(datadir + doc).read())
    all_text = " ".join(doctexts)

    sc = SpellChecker(all_text, autoparse=True)
    print('Words [peek]:', sc.words[0:10])
    print('Word Counts [peek]:', list(sc.wordcounts.items())[0:10])
    print('Unique Words [peek]:', list(sc.unique_words)[0:10])
    print('--->')

    print("{} total words found. {} Unique words".format(sc.n_words, sc.n_unique_words))
    for word in ['hobbit', 'the', 'a', 'farm', 'random', 'history', 'stalk', 'asdfasdfasdfasdf', 'stare', 'book']:
        print("Probability of {word}: ".format(word=word), str(round(sc.probability(word) * 100, 4)) + '%')
    # print("Probability of hobbit: ", str(round(sc.probability('hobbit') * 100, 2)) + '%')
    # print('probability of the word "the": ', str(round(sc.probability('the') * 100, 2)) + '%')

    for mistake in ['hobit', 'pip', 'rign', 'stlak', 'shrie', 'ownard', 'teh', 'moer', 'hlep']:
        print('Corrected "{}":'.format(mistake), sc.correction(mistake))

    print('Corrected "hobit":', sc.correction('hobit'))
    print('Correct "pip":', sc.correction('pip'))

    print('hello in words:', 'hello' in sc.unique_words)

    print("DONE!!")


if __name__ == '__main__':
    main()
