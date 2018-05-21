import re
from six import string_types
from nltk.tag import str2tuple, map_tag
import os.path

from nltk.corpus.reader.util import *
from nltk.corpus import stopwords
from nltk.corpus.reader.api import *
import string, os, ast

class UrduCorpusReader(CorpusReader):
    def split_pun(self,fileids=None): #function for punctuation removal 
        punctuations = [
        u'\u06D4',  # arabic full stop
        '.',
        u'\u061F',  # Arabic question mark
        u'\u061B', #ARABIC SEMICOLON
        u'\u066D', #ARABIC FIVE POINTED STAR
        u'\u2018' ,#LEFT SINGLE QUOTATION MARK
        u'\u2019' ,#Right Single Quotation Mark
        u'\u0027' ,#APOSTROPHE
        u'\u060c', #ARABIC COMMA
        '/',
        ':',
        ';',
        '-',
        '*',
        ')',
        '(',
        '/'
    ]
        punct_list = []
        for filepath in self.abspaths(fileids=fileids):
            print(filepath)
            text = open(filepath, 'r').read()
            text =re.sub(r'\b\d+\b\s', '', text)     #to remove digits and   if not line.strip():
            s1 = ''.join(ch for ch in text if ch not in punctuations )
            print(s1)
        return s1

    def words(self,files): #function for word tokenization
        """
        List of words, one per line.  Blank lines are ignored.
        """
        words_list = []
        files = files.replace('\n',' ')
        words_list = files.split(' ')
        print( words_list)        #printing the words after tokenization 

        return words_list
    
    def remove_stopwords(self,ifile):
        processed_word_list = []
        stopword = stopwords.words("urdu")
        words = ifile
        for word in words:
            if word not in stopword:
                processed_word_list.append(word)
            else:
                processed_word_list.append('*')
        print(processed_word_list)
        return processed_word_list
    
    
    def raw(self, fileids=None):
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, string_types):
            fileids = [fileids]

        return concat([self.open(f).read() for f in fileids])
