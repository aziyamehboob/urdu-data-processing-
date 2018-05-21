import subprocess
import os.path
from urdu_corpus_reader import UrduCorpusReader

from lda2 import *
from lsi import *
import logging
import re
import operator
import json
import gensim
import numpy as np
from gensim import corpora, models
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from collections import defaultdict
from lda1 import default_lda_model

if '__main__' == __name__:

    corpus_root = os.path.abspath('../raw_urdu_data')
    wordlists = UrduCorpusReader(corpus_root, '.*')
    noun = []
    results = {}   # AM: results dict should have been defined outside the loop otherwise it gets
                   # created on each loop iteration

    # AM: processing all the corpus files at a time is going to become memory heavy as the size of
    # the corpus increases. Perhaps we should start storing results of processed files to disk?
    for infile in (wordlists.fileids()):
        print(infile)
        cmd = "./tree-tagger -token urdu.par <<EOF\n{}\nEOF\n".format(
            '\n'.join(wordlists.words(infile))
        )
        cp = subprocess.run(cmd, shell=True, check=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        tagged_words = []
        for line in cp.stdout.decode('utf-8').split('\n'):
            if line.strip() == '':
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            tagged_words.append(parts)
        #print(tagged_words)
        results[infile] = [{'token' : token, 'tagged_result': pos} for token, pos in tagged_words]
        #only noun 
        texts = [word for word, pos in tagged_words if (pos == 'NN' )]

        # AM: why are we appending all nouns, how would we classify different categories this way?
        # We're processing all files and using all nouns. Shouldn't we instead pass nouns by
        # classification category? For example all nouns related to sports news etc.
        noun.append(texts)  

    #print(noun)
    #remove words that appear only once 
    all_tokens = sum(noun, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    texts = [[word for word in text if word not in tokens_once] for text in noun]
    #create dictionary
    dictionary= corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    #################apply lda model################

    print("\n====results from lsi====\n")
    result = lsi_model(corpus,dictionary)
    print("")
    print("\n=====resluts from default lda model=======\n")
    result =  default_lda_model(corpus,dictionary)
    print("")
    print ("\n=====result from coherence model=====\n")
    result =  coherence_lda_model(corpus,dictionary)
        #noun = []
        #word = [word for word in tagged_words[token] if word[pos] in ['NN']]
       # print(words)
    

       
