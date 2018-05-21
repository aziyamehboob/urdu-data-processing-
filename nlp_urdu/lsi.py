import gensim
import numpy as np
from gensim import corpora, models
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel

def lsi_model(corpus,dictionary):
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
    lsi.print_topics()
    for i in lsi.print_topics():
        print( i)
