
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
import glob
from math import log
from collections import defaultdict
import itertools
import numpy as np
import string, os, ast


def load_corpus():
    corpus_root = os.path.abspath('../multiwords/input_data')
    mycorpus = nltk.corpus.reader.TaggedCorpusReader(corpus_root,'.*')
    return mycorpus.tagged_sents()

def chunk_sents(tagg_sents, pos_pattern):
    chunk_freq_dict = defaultdict(int)
    chunker = nltk.RegexpParser(pos_pattern)
    for sent in tagg_sents:
        #print(type(sent))
        #print(sent)
        for chk in chunker.parse(sent).subtrees():
            if str(chk).startswith('(NP'):
                f = open("terms.txt", "a")   
                phrase = chk.__unicode__()[4:-1]
               # print(phrase)
                f.write(phrase + "\n")
                f.close()
                if '\n' in phrase:
                    phrase = ' '.join(phrase.split())
                    #print(phrase)
                chunk_freq_dict[phrase] += 1
    #print(chunk_freq_dict)
    return chunk_freq_dict

def min_freq_filter(chunk_freq_dict, min_freq):
    chunk_freq_dict = \
        dict([p for p in chunk_freq_dict.items() if p[1] >= min_freq])
    #print(chunk_freq_dict)
    return chunk_freq_dict


def remove_str_postags(tagged_str):
    stripped_str = ' '.join([w.rsplit('/', 1)[0] for w in tagged_str.split()])
    #print(stripped_str)
    return stripped_str


def remove_dict_postags(chunk_freq_dict):
    new_dict = {}
    new_list = []
    for phrase in chunk_freq_dict.keys():
        #print(phrase)
        new_str = remove_str_postags(phrase)
        #####to store it in new list#######
        new_list.append(new_str)
        #print(new_list)
        new_dict[new_str] = chunk_freq_dict[phrase]
        #print(new_dict)
    return new_dict

def build_sorted_chunks(chunk_freq_dict):
    sorted_chunk_dict = defaultdict(list)
    for phrs in chunk_freq_dict.items():
        sorted_chunk_dict[len(phrs[0].split())].append(phrs)
    for num_words in sorted_chunk_dict.keys():
        sorted_chunk_dict[num_words] = sorted(sorted_chunk_dict[num_words],
                                              key=lambda item: item[1],
                                              reverse=True)
    #print(sorted_chunk_dict)
    return sorted_chunk_dict


def calc_cvalue(sorted_phrase_dict, min_cvalue):
    cvalue_dict = {}
    triple_dict = {}  # 'candidate string': (f(b), t(b), c(b))
    max_num_words = max(sorted_phrase_dict.keys())

    # Longest candidates.
    for phrs_a, freq_a in sorted_phrase_dict[max_num_words]:
        cvalue = (1.0 + log(len(phrs_a.split()), 2)) * freq_a
        if cvalue >= min_cvalue:
            cvalue_dict[phrs_a] = cvalue
            for num_words in reversed(range(1, max_num_words)):
                for phrs_b, freq_b in sorted_phrase_dict[num_words]:
                    if set(phrs_b.split()).issubset(set(phrs_a.split())) and \
                            phrs_b in phrs_a:
                        if phrs_b not in triple_dict.keys():  # create triple
                            triple_dict[phrs_b] = (freq_b, freq_a, 1)
                        else:                                 # update triple
                            fb, old_tb, old_cb = triple_dict[phrs_b]
                            triple_dict[phrs_b] = \
                                (fb, old_tb + freq_a, old_cb + 1)

    # Candidates with num. words < max num. words
    num_words_counter = max_num_words - 1
    while num_words_counter > 0:
        for phrs_a, freq_a in sorted_phrase_dict[num_words_counter]:
            if phrs_a not in triple_dict.keys():
                cvalue = (1.0 + log(len(phrs_a.split()), 2)) * freq_a
                if cvalue >= min_cvalue:
                    cvalue_dict[phrs_a] = cvalue
            else:
                cvalue = (1.0 + log(len(phrs_a.split()), 2)) * \
                    (freq_a - ((1/triple_dict[phrs_a][2])
                               * triple_dict[phrs_a][1]))
                if cvalue >= min_cvalue:
                    cvalue_dict[phrs_a] = cvalue
            if cvalue >= min_cvalue:
                for num_words in reversed(range(1, num_words_counter)):
                    for phrs_b, freq_b in sorted_phrase_dict[num_words]:
                        if set(phrs_b.split()).issubset(set(phrs_a.split())) \
                                and phrs_b in phrs_a:
                            if phrs_b not in triple_dict.keys():  # make triple
                                triple_dict[phrs_b] = (freq_b, freq_a, 1)
                            else:                                 # updt triple
                                fb, old_tb, old_cb = triple_dict[phrs_b]
# if/else below: If n(a) is the number of times a has appeared as nested, then
# t(b) will be increased by f(a) - n(a). Frantzi, et al (2000), end of p.5.
                                if phrs_a in triple_dict.keys():
                                    triple_dict[phrs_b] = (
                                        fb, old_tb + freq_a -
                                        triple_dict[phrs_a][1], old_cb + 1)
                                else:
                                    triple_dict[phrs_b] = (
                                        fb, old_tb + freq_a, old_cb + 1)
        num_words_counter -= 1

    return cvalue_dict




def recalc_chunk_freq(domain_sents, untagged_chunk_freqs):
    corpus = ' '
    for sent in domain_sents:
        for word_tag in sent:
            corpus += word_tag[0]
            corpus += ' '
    corpus += ' '
    new_freqs = {}
    for chunk in untagged_chunk_freqs.keys():
        nchunk = ' ' + chunk + ' '
        new_freqs[chunk] = corpus.count(nchunk)
        #print(new_freqs)
    return new_freqs




    
def main(domain_corpus, pos_pattern,min_freq,min_cvalue):
    # STEP 1
    domain_sents = domain_corpus
    #print("domain_sents:", domain_sents) 
    #print("type(domain_sents):", type(domain_sents))
    # Extract matching patterns
    chunks_freqs = chunk_sents(domain_sents, pos_pattern)
    # Remove POS tags from chunks
    chunks_freqs = remove_dict_postags(chunks_freqs)
    chunks_freqs = recalc_chunk_freq(domain_sents, chunks_freqs)

    # Discard chunks that don't meet minimum frequency
    chunks_freqs = min_freq_filter(chunks_freqs, min_freq)
    
   # chunks_freqs = stoplist_filter(chunks_freqs, stoplist)

    # Order candidates first by number of words, then by frequency
    sorted_chunks = build_sorted_chunks(chunks_freqs)

    # STEP 3
    # Calculate C-value
    cvalue_output = calc_cvalue(sorted_chunks, min_cvalue)
    return cvalue_output
    #return chunks_freqs


if __name__ == '__main__':
    PATTERN = r"""
       NP: {<ADJ><NN>+}
        """
    MIN_FREQ = 1
    MIN_CVAL = -13 # lowest cval -13

    domain_corpus = load_corpus()
    candidates = main(domain_corpus, PATTERN,MIN_FREQ,MIN_CVAL)
    sorted_candidates = [(cand, score) for cand, score in sorted(candidates.items(), key=lambda x: x[1], reverse=True)]

    with open('cvalue-ADJNN+.txt', 'w') as f:
        new_cands = []
        for c in sorted_candidates:
            newc = '%.5f\t%s' % (c[1], c[0])
            new_cands.append(newc)
        f.write('\n'.join(new_cands))
    
        


