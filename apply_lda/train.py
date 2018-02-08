import logging
import sys, os
import json
import gensim
from gensim.corpora import BleiCorpus
from gensim import corpora



class Corpus(object):
    def __init__(self):
        self.lda_model_path ="models/lda_model_10_topics.lda"
        self.dictionary_path = "models/dictionary.dict"

    def get_input(self,corpus_lda):
        # remove words that appear only once
        all_tokens = sum(corpus_lda, [])
        print('all tokens:%d' % len(all_tokens))
        tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
        #print(tokens_once)
        texts = [[word for word in text if word not in tokens_once] for text in corpus_lda]
        # ##############Create Dictionary.#####################
        dictionary= corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=3, no_above=0.5)
        dictionary.compactify() 
        corpora.Dictionary.save(dictionary, self.dictionary_path)
        # ############Creates the Bag of Word corpus.###########################
        corpus = [dictionary.doc2bow(text) for text in corpus_lda]
        #print(corpus)
        print('Number of unique tokens: %d' % len(dictionary))
        print('Number of documents: %d' % len(corpus))
        corpora.MmCorpus.serialize('/tmp/corpus_file.mm', corpus)
        
        return dictionary,corpus


    def run(self,lda_model_path, corpus,num_topics, dictionary):
        filename = sys.argv[1]
        lda = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary,alpha = 0.001, chunksize=10000, passes=50,iterations=300)
        
        topics = []
        for i in range(0, lda.num_topics):
            top_cluster ={}
            print('{}\n'.format('Topic #' + str(i + 1) + ': '))
            for word, prob in lda.show_topic(i, topn=10):
                top_cluster[word] = prob
                print(top_cluster)
            
            topics.append(top_cluster)
                
        # We can also do some thing like this
        '''tops = set(lda.show_topics(10))
        top_clusters = []
        for l in tops:
            #print(l)
            top = []
            for t in str(l).split(" + "):
                top.append((t.split("*")[0], t.split("*")[1]))
            top_clusters.append(top)
        print(top_clusters)'''
                    

        #lda.save(self.lda_model_path)

        return topics
