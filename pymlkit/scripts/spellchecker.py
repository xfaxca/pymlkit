from pymlkit.nlp.spellcheck import SpellChecker

TIME_LOOKUPS = True


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

    # Instantiate a SpellChecker and parse the text.
    sc = SpellChecker(all_text, autoparse=True)  # alternatively, can call `parse_words` and `count_Words` methods separately.

    print("{} total words found. {} Unique words".format(sc.n_words, sc.n_unique_words))
    for word in ['hobbit', 'the', 'a', 'farm', 'random', 'history', 'stalk', 'asdfasdfasdfasdf', 'stare', 'book']:
        print("Probability of {word}: ".format(word=word), str(round(sc.probability(word) * 100, 4)) + '%')
    for mistake in ['hobit', 'pip', 'rign', 'stlak', 'shrie', 'ownard', 'teh', 'moer', 'hlep']:
        print('Corrected "{}":'.format(mistake), sc.correction(mistake))

    loop = True
    while loop:
        word = input("Please enter a word to spell check").lower().strip()
        if word in ['exit', 'quit']:
            print("Goodbye.")
            loop = False
        elif word != '':
            from time import time
            t0 = time()
            print("Probability of {word}: ".format(word=word), str(round(sc.probability(word) * 100, 4)) + '%')
            if TIME_LOOKUPS:
                print("Total time: {}ms".format(round((time() - t0) * 1000, 2)))
            print('Best suggestions to correct "{}":'.format(word), sc.correction(word, n=5))
        else:
            print("Your input was empty. Please try again or type 'quit' or 'exit' to exit.")


if __name__ == '__main__':
    main()
