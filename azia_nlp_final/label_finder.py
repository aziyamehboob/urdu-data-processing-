import nltk
from nltk.collocations import BigramCollocationFinder
from toolz.itertoolz import get
from toolz.functoolz import partial


class BigramLabelFinder(object):
    def __init__(self, min_freq=10):
        """
        min_freq: int
            minimal frequency for the label to be considered
        """

        self.score_func = nltk.collocations.BigramAssocMeasures().pmi

        self._min_freq = min_freq

    def find(self, docs, top_n, strip_tags=True):
        """
        Parameter:
        ---------------

        docs: list of tokenized documents

        top_n: int
            how many labels to return

        strip_tags: bool
            whether return without the POS tags or not

        Return:
        ---------------
        list of tuple of str: the bigrams
        """
        # if apply pos constraints
        # check the pos properties

        score_func = self.score_func

        finder = BigramCollocationFinder.from_documents(docs)
        finder.apply_freq_filter(self._min_freq)

        bigrams = finder.nbest(score_func,
                               top_n)
        return bigrams
