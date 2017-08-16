import os.path
from urdu_corpus_reader import UrduCorpusReader
from stop_words import remove_urdu_stopwords


if '__main__' == __name__:

    corpus_root = os.path.abspath('../raw_urdu_data')
    wordlists = UrduCorpusReader(corpus_root, '.*')

    print("Loaded corpus with file IDs: ")
    print(wordlists.fileids())

    sample_id = wordlists.fileids()[10]

    print("\n\nSENTeNCES\n===============\n")
    idx = 1
    for s in wordlists.sents(sample_id):
        print("\nSentence {}\n----------------\n".format(idx))
        print(s)
        idx += 1

    print("Words from file: " + sample_id)
    for w in wordlists.words(sample_id):
        print(w, end='   ')

    # URDU STOP WORDS REMOVAL
    stopwords_corpus = UrduCorpusReader('./data', ['stopwords-ur.txt'])    
    stopwords = stopwords_corpus.words()
    # print(stopwords)

    words = wordlists.words(sample_id)
    finalized_words = remove_urdu_stopwords(stopwords, words)
    print("\n==== WITHOUT STOPWORDS ===========\n")
    print(finalized_words)
