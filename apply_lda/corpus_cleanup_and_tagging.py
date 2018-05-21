import subprocess
import os.path
from urdu_corpus_reader import UrduCorpusReader
import re
#from lda2 import *
#from lsi import *
import logging
import re
import operator
import json
import gensim
import numpy as np
from gensim import corpora, models
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from collections import defaultdict
#from lda1 import default_lda_model

newpath = 'new_output' 
if not os.path.exists(newpath):
    os.makedirs(newpath)




if '__main__' == __name__:

    corpus_root = os.path.abspath('../scrap/urdu_results')
    wordlists = UrduCorpusReader(corpus_root, '.*')
    noun = []

    for infile in (wordlists.fileids()):
        print(infile)
        open('/tmp/tree_tagger_temp_input.txt', 'w').write('\n'.join(wordlists.words(infile)))
        cmd = "./tree-tagger -token urdu.par /tmp/tree_tagger_temp_input.txt"
        print(cmd)
        cp = subprocess.run(cmd, shell=True, check=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        tagged_words = []
        for line in cp.stdout.decode('utf-8').split('\n'):
            if line.strip() == '':
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            tagged_words.append(parts)
        text = " ".join(w+"/"+t for w,t in tagged_words)
        output = os.path.basename(infile)
        open('new_output/' + output, 'w').write(str(text+"\n"))
    #for tagged_sent in tagged_words:
            #print(tagged_sent)

        #only noun 
    '''  texts = [word for word, pos in tagged_words if (pos == 'NN' )]

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
    result =  coherence_lda_model(corpus,dictionary)'''
    

       
