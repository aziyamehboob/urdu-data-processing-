from nltk.corpus import PlaintextCorpusReader
import os.path


if '__main__' == __name__:

    corpus_root = os.path.abspath('./raw_urdu_data')
    wordlists = PlaintextCorpusReader(corpus_root, '.*')

    print("Loaded corpus with file IDs: ")
    print(wordlists.fileids())

    sample_id = wordlists.fileids()[0]
    print("Words from file: " + sample_id)
    for w in wordlists.words(sample_id):
        print(w, end='   ')

    print("\n\nSENTeNCES\n===============\n")
    idx = 1
    for s in wordlists.sents('entertainment-40150981.txt'):
        print("\nSentence {}\n----------------\n".format(idx))
        print(s)
        idx += 1
