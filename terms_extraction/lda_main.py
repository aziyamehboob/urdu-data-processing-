import nltk
from math import log
from collections import defaultdict
import os.path
from lsi import *
import gensim
import numpy as np
from gensim import corpora, models
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from collections import defaultdict
from lda1 import default_lda_model
from collections import Counter
import re

if __name__ == '__main__':
    corpus_root = os.path.abspath('../terms_extraction/tagged_data')
    mycorpus = nltk.corpus.reader.TaggedCorpusReader(corpus_root,'.*')
    noun=[]
    count_freq = defaultdict(int)
 
    for infile in (mycorpus.fileids()):
        print(infile)
    for i in (mycorpus.tagged_sents()):
         texts = [word for word, pos in i  if (pos == 'NN' )]
         noun.append(texts)
    for doc in noun:     # count all token
             for token in doc:
                 count_freq[token] += 1
    print('Totel Number of Nouns: %d' % len(count_freq))
    dictionary = corpora.Dictionary(noun)
    #vocabulary
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in noun]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    print("\n=====resluts from default lda model=======\n")
        #################apply lda model################

    print("\n====results from lsi====\n")
    result = lsi_model(corpus,dictionary)
    print("")
    print("\n=====resluts from default lda model=======\n")
    result =  default_lda_model(corpus,dictionary)
    print("")
    print ("\n=====result from coherence model=====\n")
    result =  coherence_lda_model(corpus,dictionary)

    
        
