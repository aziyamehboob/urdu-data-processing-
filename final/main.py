import os.path
import re
from process_text import UrduCorpusReader

if '__main__' == __name__:
    
    word = ' '
    corpus_root = os.path.abspath('../test_data')
    wordlists = UrduCorpusReader(corpus_root, '.*')
    print("Loaded corpus with file IDs: ")
    print(wordlists.fileids())
    list1 = wordlists.fileids()
    for infile in (wordlists.fileids()):
        #print(infile)
        word = wordlists.split_pun(infile) 
        word= "".join([s for s in word.strip().splitlines(True) if s.strip("\r\n").strip()])
        pun = wordlists.words(word)
        text=wordlists.remove_stopwords(pun)
        #while '*' in text: text.remove('*')  
        #print(text)
