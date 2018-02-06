"""
Natural Language Toolkit: Urdu Language POS-Tagged (not yet) Corpus Reader
"""

from six import string_types
import re
import string
from nltk.tag import str2tuple, map_tag
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.api import CorpusReader


class UrduCorpusReader(CorpusReader):

    SENT_SEPS = [
        u'\u06D4',  # arabic full stop
        '.',
        u'\u061F'  # Arabic question mark
    ]

    PUNCTUATIONS = [
        '.', '/', ':', ';', '-', '*', ')', '(', '/', '\n', '\r', '`', '%', '&', '>',' ــــ ','<', '|','[',']','{','}','_',
        u'\u06D4',   # arabic full stop
        u' \u0621 ', #arabic hamza
        u'\u061F',   # Arabic question mark
        u'\u061B',   # ARABIC SEMICOLON
        u'\u066D',   # ARABIC FIVE POINTED STAR
        u'\u2018',   # LEFT SINGLE QUOTATION MARK
        u'\u2019',   # Right Single Quotation Mark
        u'\u0027',   # APOSTROPHE
        u'\u060c',    # ARABIC COMMA
        u'\u200b',    #ZERO_WIDTH_SPACE
       u' \u200f' ,  ## RIGHT_TO_LEFT_MARk
       u'\u200c',  # ZERO_WIDTH_NON_JOINER
      u'\u200d',   # ZERO_WIDTH_JOINER
      u'\u200e',# LEFT_TO_RIGHT_MARK
      '\uFEFF',   # BYTE_ORDER_MARK
      '\uFFFE' #BYTE_ORDER_MARK_2
    ]

    def words(self, fileids=None):
        """
        List of words, one per line.  Blank lines are ignored.
        """
        words_list = []
        us = ''
        in_urdu = False

        for filepath in self.abspaths(fileids=fileids):
            print(filepath)
            data = open(filepath, 'r').read()
            data= ''.join(filter(lambda x: x==' ' or x not in string.printable , data))
            data = re.sub('[a-zA-Z0-9]+', '', data)
            for p in self.PUNCTUATIONS:
                data = data.replace(p, ' ')
                #data = ''.join(i for i in data if not  i.isalpha())
               # print(data)
            words_list.extend([w for w in data.split(' ') if w])
            #print (words_list)

        return words_list


    def sents(self, fileids=None):
        sents_list = []
        current_sent = ''
        for filepath in self.abspaths(fileids=fileids):
            print(filepath)
            data = open(filepath, 'r').read()
            for ch in data:
                if ch in ['\r', '\n']:  # ignore new lines
                    continue

                if ch in self.SENT_SEPS:
                    if current_sent:
                        sents_list.append(current_sent)

                    current_sent = ''
                    continue

                current_sent += ch

        if current_sent:
            sents_list.append(current_sent)

        return sents_list

    def raw(self, fileids=None):
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]

        return concat([self.open(f).read() for f in fileids])
