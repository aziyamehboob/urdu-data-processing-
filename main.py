import os.path
from corpus_reader import UrduCorpusReader



if '__main__' == __name__:
    
    newpath = 'output_data' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)


    corpus_root = os.path.abspath('../test_data')
    wordlists = UrduCorpusReader(corpus_root, '.*')

    print("Loaded corpus with file IDs: ")
    print(wordlists.fileids())
    list1 = wordlists.fileids()
    print("\n\nSENTeNCES\n===============\n")
    idx = 0
    for infile in (wordlists.fileids()):
        #print(infile)
        for s in wordlists.sents(infile):
            rd = s
            print(rd)
            output = os.path.basename(infile)
            open('output_data/' + output, 'w').write(rd)
            #############################################################
            
for i in (list1):
    sample_id = i
    print("Words from file: " + sample_id)
    for w in wordlists.words(sample_id):
        s = w
        #print(w, end='   ')
