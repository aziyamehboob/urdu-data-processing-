"""
LDA automatic labeling.

Derived from chowmein automatic labeling code but using cvalue based corpus
processing and made the code python 3 compatible.
"""

import argparse
import codecs
import pickle
from corpus_processor import get_lda_input_from_corpus_folder
import lda
import numpy as np

from sklearn.feature_extraction.text import (CountVectorizer
                                             as WordCountVectorizer)
from text import LabelCountVectorizer
from label_finder import BigramLabelFinder
from label_ranker import LabelRanker
from pmi import PMICalculator


CORPUS_PATH = './dataset/TRAIN'
STOP_WORDS_FILES = ['./data/multi_stopwords-ur.txt', './data/stopwords-ur.txt']

def create_parser():
    parser = argparse.ArgumentParser(
        description="Command line interface that perform topic modeling " +
        " and topic model labeling")

    # corpus path and preprocessing
    parser.add_argument('--n_cand_labels', type=int, default=100,
                        help='Number of candidate labels to take')
    parser.add_argument('--label_min_df', type=int, default=5,
                        help='Minimum document frequency requirement for candidate labels')

    # LDA
    parser.add_argument('--lda_random_state', type=int, default=12345,
                        help='Random state for LDA modeling')
    parser.add_argument('--lda_n_iter', type=int, default=400,
                        help='Iteraction number for LDA modeling')
    parser.add_argument('--n_topics', type=int, default=6,
                        help='Number of topics')
    parser.add_argument('--n_top_words', type=int, default=15,
                        help='Number of topical words to display for each topic')

    # Topic label
    parser.add_argument('--n_labels', type=int, default=8,
                        help='Number of labels displayed per topic')

    return parser


def load_stopwords(files):
    """Load stopwords from list of filenames."""
    ret = []
    for fname in files:
        with codecs.open(fname,  'r', 'utf8') as f:
            ret.extend([i for i in map(lambda s: s.strip(), f.readlines())])
    return ret


def get_topic_labels(n_topics,
                     n_top_words,
                     n_cand_labels, label_min_df,
                     n_labels,
                     lda_random_state,
                     lda_n_iter):
    """
    Refer the arguments to `create_parser`
    """
    print("Loading docs and preprocessing (cvalue etc) for lda input...")
    # docs = get_lda_input_from_corpus_folder(CORPUS_PATH)
    # docs = load_line_corpus(corpus_path)
    docs = pickle.load(open('./data/lda_input_docs_finalized.pickle', 'rb'))

    print("Generate candidate bigram labels(with POS filtering)...")
    finder = BigramLabelFinder(min_freq=label_min_df)
    cand_labels = finder.find(docs, top_n=n_cand_labels)

    print("Collected {} candidate labels".format(len(cand_labels)))

    print("Calculate the PMI scores...")

    pmi_cal = PMICalculator(
        doc2word_vectorizer=WordCountVectorizer(
            min_df=5,
            stop_words=load_stopwords(STOP_WORDS_FILES)),
        doc2label_vectorizer=LabelCountVectorizer())

    pmi_w2l = pmi_cal.from_texts(docs, cand_labels)

    print("Topic modeling using LDA...")
    model = lda.LDA(n_topics=n_topics, n_iter=lda_n_iter,
                    random_state=lda_random_state)
    model.fit(pmi_cal.d2w_)

    print("\nTopical words:")
    print("-" * 20)
    for i, topic_dist in enumerate(model.topic_word_):
        top_word_ids = np.argsort(topic_dist)[:-n_top_words:-1]
        topic_words = [pmi_cal.index2word_[id_]
                       for id_ in top_word_ids]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    ranker = LabelRanker(apply_intra_topic_coverage=False)

    return ranker.top_k_labels(topic_models=model.topic_word_,
                               pmi_w2l=pmi_w2l,
                               index2label=pmi_cal.index2label_,
                               label_models=None,
                               k=n_labels)




if __name__ == '__main__':

    parser = create_parser()

    args = parser.parse_args()
    labels = get_topic_labels(n_topics=args.n_topics,
                              n_top_words=args.n_top_words,
                              n_cand_labels=args.n_cand_labels,
                              label_min_df=args.label_min_df,
                              n_labels=args.n_labels,
                              lda_random_state=args.lda_random_state,
                              lda_n_iter=args.lda_n_iter)

    print("\nTopical labels:")
    print("-" * 20)
    for i, labels in enumerate(labels):
        print(u"Topic {}: {}\n".format(
            i,
            ', '.join(map(lambda l: ' '.join(l), labels))
        ))
