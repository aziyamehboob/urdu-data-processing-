from corpus_processor import get_lda_input_from_corpus_folder
from lda_model import LDAModel
from pprint import pprint
from collections import Counter
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel


if __name__ == '__main__':
    #limit=40
    #start=2
    #step=2
    obj = LDAModel()
    lda_input = get_lda_input_from_corpus_folder('./dataset/TRAIN')
    output = obj. extract_words(lda_input)
    lm, top_topics =obj.lda_train(lda_input)
    print(top_topics[:5])
    #print("show topics",lm.show_topics(formatted=False))
    pprint([lm.show_topic(topicid, topn=12) for topicid, c_v in top_topics[:8]])
    
    
    #lda_lsi_topics = [[word for word, prob in lm.show_topic(topicid)] for topicid, c_v in top_topics]
   # print ("topic of lda_lsi", lda_lsi_topics)
   

   # model_list, coherence_values = obj.compute_coherence_values(dictionary=dic,corpus=corpus, texts=lda_input,  start=2, limit=40, step=2)
   # x = range(start, limit, step)
    #for m, cv in zip(x, coherence_values):
        #print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    #words = obj.extract_words(lda_input)
    #print(lda_input)

    #dic, corp, mod = obj.lda_train(lda_input)
    #print(top_topics[:5])
    #pprint([lm.show_topic(topicid) for topicid, c_v in top_topics[:10]])

    
    #topics = mod.print_topics(num_words=10)
    #for topic in topics:
        #print(topic)
