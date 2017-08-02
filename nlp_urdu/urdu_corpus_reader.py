"""
Natural Language Toolkit: Urdu Language POS-Tagged (not yet) Corpus Reader
"""

from six import string_types

from nltk.tag import str2tuple, map_tag

from nltk.corpus.reader.util import *
from nltk.corpus.reader.api import *


class UrduCorpusReader(CorpusReader):

    SENT_SEPS = [
        u'\u06D4',  # arabic full stop
        '.',
        u'\u061F'  # Arabic question mark
    ]

    def words(self, fileids=None):
        """
        List of words, one per line.  Blank lines are ignored.
        """
        words_list = []
        for filepath in self.abspaths(fileids=fileids):
            print(filepath)
            data = open(filepath, 'r').read()
            words_list = data.split(' ')

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
