import os
from gensim import corpora, models
import nltk
from pprint import pprint
from gensim.models import LdaModel, CoherenceModel
import operator
import json
from collections import Counter


class LDAModel(object):

    DICTIONARY_FILE = "exports/ta.dict"
    BOW_FILE = "exports/ta_bow.mm"
    LDA_MODEL_FILE = "exports/ta_model.lda"
    LDA_TOPICS_FILE = os.path.abspath('./exports/train')
    Word_count = os.path.abspath('./exports/count')
    TEMP_CORPUS_FOLDER = 'test_corpus'
    def extract_words(self ,all_doc):
        with open(self.Word_count, "wt") as outf:
            all_words = []
            for d in all_doc:
                for w in d:
                    all_words.append(w)# count all words and the select words between some range 
            word_count = Counter(all_words)
            print("length of words",len(word_count))
            for word  in sorted(word_count, key=word_count.get, reverse=True ):
                outf.write('{}|{}\n'.format(word,word_count[word]))  
            #print("counting all the word:",word_count)
            return word_count
  

    def save_topics(self, model):
        with open(self.LDA_TOPICS_FILE, "wt") as outf:
            # ---------- write each topic and words' contribution
            topics = model.show_topics(num_topics=-1, log=False, formatted=True)
            for topic in topics:
                # topic[0]: topic number
                # topic[1]: topic description
                outf.write("\n############# TOPIC {} #############\n".format(topic[0]))
                outf.write(topic[1]+"\n")
            # ---------- words statistics in all topics
            outf.write("\n\n\n****************** KEY WORDS ******************\n")
            topics = model.show_topics(num_topics=-1, log=False, formatted=False)
            keywords = (word for (_,words) in topics for (word,score) in words)
            fdist = nltk.FreqDist(keywords)
            for index,(w,c) in enumerate( fdist.most_common(100) ):
                outf.write("{}-th keyword: <{},{}>\n".format(index+1,w,c))

    def lda_train(self, documents,iterations=20):
        print("length of documents",len(documents))
        # create dictionary
        num_topics = 7
        dictionary= corpora.Dictionary(documents)
        dictionary.filter_extremes(no_below=20, no_above=0.3)
        #dictionary.filter_extremes(keep_n=10000)
        dictionary.compactify()
        dictionary.save(self.DICTIONARY_FILE)  # store the dictionary, for future reference
        print ("============ Dictionary Generated and Saved ============")
        ############# Create Corpus##########################
        # => can be safely removed
        corpus = [dictionary.doc2bow(text) for text in documents]
        print('Number of unique tokens: %d' % len(dictionary))
        print('Number of documents: %d' % len(corpus))

        #############create LDA model ##########################
        result = []
        for _ in range(iterations):
            print("Iteration {}/{}".format(_ + 1, iterations))
            model = models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=8,chunksize=10000, passes=30,iterations=300)
            coherence_values = {}
            for n, topic in model.show_topics(num_topics=-1, num_words=12,formatted=False):
                topic = [word for word, _ in topic]
                cm = CoherenceModel(topics=[topic], texts=documents, dictionary=dictionary, window_size=10)
                coherence_values[n] = cm.get_coherence()
            top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
            result.append((model, top_topics))
        return max(result, key=lambda _: _[1][0][1])

    
    # just to check the number of topics 
    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model  = models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics,chunksize=10000, passes=30,iterations=100)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values
    
    
    def lda_test(self, new_documents):

        if not os.path.exists(self.TEMP_CORPUS_FOLDER):
            os.makedirs(self.TEMP_CORPUS_FOLDER)

        old_dict = corpora.Dictionary.load(self.DICTIONARY_FILE)
        # First update the dict with new documents here
        old_dict.add_documents(new_documents)
        test_corpus = [old_dict.doc2bow(text) for text in new_documents]
        trained_ldamodel = models.LdaModel.load(self.LDA_MODEL_FILE)
        trained_ldamodel.update(test_corpus)
        # LDAModel.save_topics(trained_ldamodel,LdaTopicsFile)
        topics = trained_ldamodel.print_topics(num_words=7)
        for topic in topics:
            print(topic)
