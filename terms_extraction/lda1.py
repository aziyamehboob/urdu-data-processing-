import gensim
import numpy as np
from gensim import corpora, models
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
def default_lda_model(corpus,dictionary):
    lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, \
        alpha = 0.001, chunksize=10000, passes=50,iterations=100)
    # Prints the topics.
    for top in lda.print_topics():
        print( top)
        

