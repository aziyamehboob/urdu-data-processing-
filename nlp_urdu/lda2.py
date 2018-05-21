import gensim
import numpy as np
from gensim import corpora, models
from pprint import pprint
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
def coherence_lda_model(corpus,dictionary):
    num_topics = 10
    chunksize = 2000
    passes = 40
    iterations = 50
    eval_every = None  # Don't evaluate model perplexity, takes too much time.
    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                       alpha='auto', eta='auto', \
                       iterations=iterations, num_topics=num_topics, \
                       passes=passes, eval_every=eval_every)
    top_topics = model.top_topics(corpus, num_words=20)
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    pprint(top_topics)
