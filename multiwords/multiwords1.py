
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
import glob
from collections import defaultdict
import itertools
import numpy as np
import string, os, ast


def load_corpus():
    corpus_root = os.path.abspath('../out1_data')
    mycorpus = nltk.corpus.reader.TaggedCorpusReader(corpus_root,'.*')
    return mycorpus.tagged_sents()

def chunk_sents(tagg_sents, pos_pattern):
    chunk_freq_dict = defaultdict(int)
    chunker = nltk.RegexpParser(pos_pattern)
    for sent in tagg_sents:
        for chk in chunker.parse(sent).subtrees():
            if str(chk).startswith('(NP'):
                phrase = chk.__unicode__()[4:-1]
                print(phrase)
                if '\n' in phrase:
                    phrase = ' '.join(phrase.split())
                    print(phrase)
                chunk_freq_dict[phrase] += 1
    print(chunk_freq_dict)
    return chunk_freq_dict
    #chunked = []
    #for s in tagg_sents:
        #print(s)
        #chunked.append(chunker.parse(s))
    #print(chunked)
    
def main(domain_corpus, pos_pattern):
    # STEP 1
    domain_sents = domain_corpus
    #print("domain_sents:", domain_sents) 
    #print("type(domain_sents):", type(domain_sents))
    # Extract matching patterns
    chunks_freqs = chunk_sents(domain_sents, pos_pattern)
    return chunks_freqs


if __name__ == '__main__':
    PATTERN = r"""
       NP: {<ADJ><NN>+} 
        """

    domain_corpus = load_corpus()
    candidates = main(domain_corpus, PATTERN)
