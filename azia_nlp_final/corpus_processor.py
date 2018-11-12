import os
from math import log
from collections import Counter, defaultdict, OrderedDict

import nltk


class CorpusProcessor(object):

    PATTERN = r"""
    NP: {<NN><NN>+}
    {<ADJ><NN>+}
    {<NN><CC><NN>}
    """
    MIN_FREQ = 1
    MIN_CVAL = -13 # lowest cval -13

    FILTER_MIN_WORD_COUNT = 1
    FILTER_MAX_WORD_COUNT = 200

    def __init__(self, corpus_root):
        # corpus_root = os.path.abspath('../multiwords/Data_set/category-sports/train_sports')
        # corpus_root = os.path.abspath('../multiwords/Data_set/category-'+sys.argv[1]+'/train')

        self.corpus = nltk.corpus.reader.TaggedCorpusReader(corpus_root,'.*')
        self.word_count_by_document = None
        self.phrase_frequencies = None
        self.document_wise_text = {}
        self.cvalue_term_frequencies = {}
        self.stopwords = open('data/stopwords-ur.txt', 'r').read().split()
        self.stopwords_multi = open('data/multi_stopwords-ur.txt', 'r').read()

    def do_word_count_by_document(self):
        d= {}
        words = []

        for fileid in self.corpus.fileids():

            words = self.corpus.tagged_words(fileid)

            all_words = [word for word,pos in words ]
            just_words =[word for word,pos in words if pos =='NN' or pos=='ADJ']

            d[fileid] = Counter(just_words)

            self.document_wise_text[fileid] = ' ' + ' '.join(all_words) + ' '

        self.word_count_by_document = d
        #print("word count  of document :", self.word_count_by_document)

    def calculate_phrase_frequencies(self):
        """
        extract the sentence chunks according to PATTERN and calculate
        the frequency of chunks with pos tags
        """

        # pdb.set_trace()
        chunk_freq_dict = defaultdict(int)
        chunker = nltk.RegexpParser(self.PATTERN)

        for sent in self.corpus.tagged_sents():

            sent = [s for s in sent if s[1] is not None]

            for chk in chunker.parse(sent).subtrees():

                if str(chk).startswith('(NP'):

                    phrase = chk.__unicode__()[4:-1]
                    #print(phrase)

                    if '\n' in phrase:
                        phrase = ' '.join(phrase.split())
                        #print(phrase)

                    just_phrase = ' '.join([w.rsplit('/', 1)[0] for w in phrase.split(' ')])
                    #print(just_phrase)

                    chunk_freq_dict[just_phrase] += 1

        # trying to do the same thing as recalc_chunk_freq here so we don't redo and waste
        # processing time
        full_corpus_text = ' ' + ' '.join(self.document_wise_text.values()) + ' '
        new_freqs = {}
        for chunk in chunk_freq_dict.keys():
            nchunk = ' ' + chunk + ' '
            new_freqs[chunk] = full_corpus_text.count(nchunk)

        self.phrase_frequencies = new_freqs
        #self.phrase_frequencies = chunk_freq_dict
        #print(self.phrase_frequencies)

    def apply_frequency_filter(self):
        self.phrase_frequencies = \
            dict([p for p in self.phrase_frequencies.items() if p[1] >= self.MIN_FREQ])

    def build_sorted_chunks(self):
        "sort the chunks according to their frequencies "

        sorted_chunk_dict = defaultdict(list)

        for phrs in self.phrase_frequencies.items():  # storing according to number of words
            sorted_chunk_dict[len(phrs[0].split())].append(phrs)

        for num_words in sorted_chunk_dict.keys():
            sorted_chunk_dict[num_words] = sorted(sorted_chunk_dict[num_words],
                                                  key=lambda item: item[1],
                                                  reverse=True)
        #print(sorted_chunk_dict)
        return sorted_chunk_dict

    def calculate_cvalue(self, sorted_phrase_dict, min_cvalue): # pylint: disable=R0912,R0914
        "calculate C-value for the pattern which is mentioned say <NOUN><NOUN>+"

        cvalue_dict = OrderedDict()
        triple_dict = {}  # 'candidate string': (f(b), t(b), c(b))
        max_num_words = max(sorted_phrase_dict.keys())

        # Longest candidates.
        for phrs_a, freq_a in sorted_phrase_dict[max_num_words]: # pylint: disable=R1702
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
        while num_words_counter > 0:  # pylint: disable=R1702
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

        self.cvalue_term_frequencies = cvalue_dict
      #  print(self.cvalue_term_frequencies)

    def do_cvalue(self):

        self.calculate_phrase_frequencies()
        self.apply_frequency_filter()

        # Order candidates first by number of words, then by frequency
        sorted_chunks = self.build_sorted_chunks()
       # print(sorted_chunks)

        # STEP 3
        # Calculate C-value
        self.calculate_cvalue(sorted_chunks, self.MIN_CVAL)

    def calculate_cvalue_term_freqs_by_document(self):
        """
        Calculate document wise count for cvalue terms present in self.cvalue_term_frequencies
        """

        d = {}

        for fileid, doc_text_orig in self.document_wise_text.items():
            doc_text = doc_text_orig
            #print(doc_text)
            d_freqs = {}
            for cv_term in sorted(self.cvalue_term_frequencies.keys(), key=len, reverse=True):
               # print(cv_term)
                tc = doc_text.count(cv_term)
                doc_text = doc_text.replace(cv_term, '')
                if tc > 0:
                    d_freqs[cv_term] = tc
                    #print(d_freqs)

            d[fileid] = d_freqs

        self.cvterm_count_by_document = d
        #print(self.cvterm_count_by_document)

    def adjust_word_counts_wrt_cvterm_counts(self):
        """
        Subtract from word counts the count of cvalues terms if the word is part of the term
        """

        #print("word count in comparision code:",self.word_count_by_document)
        for doc_id, cvtd in self.cvterm_count_by_document.items():

            #print("doc id and multword terms",doc_id,cvtd)
            for cv_term, tc in cvtd.items():
                for term_word in cv_term.split(' '):
                    if term_word in self.word_count_by_document[doc_id]:
                        self.word_count_by_document[doc_id][term_word] -= tc
            #print(self.word_count_by_document)
            #self.word_count_by_document[doc_id][term_word]

    def should_filter(self, item, item_count, stop_words_type):

        if item_count < self.FILTER_MIN_WORD_COUNT or \
            item_count > self.FILTER_MAX_WORD_COUNT:

            return True

        if 'single' == stop_words_type and item in self.stopwords:
            return True
        elif 'multi' == stop_words_type and item in self.stopwords_multi:
            return True

        return False

    def get_lda_input(self):
        """
        Make input list of lists for lda by combining self.word_count_by_document and
        self.cvterm_count_by_document and placing each word or term the number of times of
        it's count
        """

        lda_input = []

        for fileid, dwc in self.word_count_by_document.items():
            doc_word_list = []
            cvdc = self.cvterm_count_by_document[fileid]

            for t, tc in cvdc.items():
                if not self.should_filter(t, tc, 'multi'):
                    for _ in range(tc):
                        doc_word_list.append(t)

            for w, wc in dwc.items():
                if not self.should_filter(w, wc, 'single'):
                    for _ in range(wc):
                        doc_word_list.append(w)

            lda_input.append(doc_word_list)


        return lda_input


def get_lda_input_from_corpus_folder(corpus_folder):

    corpus_root = os.path.abspath(corpus_folder)
    print(corpus_root)
    cp = CorpusProcessor(corpus_root)
    print(' ... doing word count by document')
    cp.do_word_count_by_document()
    print(' ... doing cvalue')
    cp.do_cvalue()
    print(' ... calculating cvalue term frequencies by document')
    cp.calculate_cvalue_term_freqs_by_document()
    print(' ... adjusting word counts w.r.t cvalue term counts')
    cp.adjust_word_counts_wrt_cvterm_counts()
    print(' ... getting final input for lda')
    lda_input = cp.get_lda_input()

    return lda_input
