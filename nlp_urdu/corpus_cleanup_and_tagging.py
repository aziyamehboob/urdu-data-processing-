import subprocess
import os.path
from urdu_corpus_reader import UrduCorpusReader


if '__main__' == __name__:

    corpus_root = os.path.abspath('../raw_urdu_data')
    wordlists = UrduCorpusReader(corpus_root, '.*')

    # print("Loaded corpus with file IDs: ")
    # print(wordlists.fileids())

    # Note: We're just taking the 11th file of the corpus for testing
    # for actual processing we might want to process all files.
    sample_id = wordlists.fileids()[10]

    # for w in wordlists.words(sample_id):
    #     print(w, end='   ')

    # ./tokenize-urdu.pl test.txt | ./tree-tagger urdu.par -token
    # انھیں طالبان نے لڑکیوں کی تعلیم کے بارے میں آواز اٹھانے پر سر میں گولی مار کر شدید زخمی کر دیا تھا

    cmd = "./tree-tagger -token urdu.par <<EOF\n{}\nEOF\n".format(
        '\n'.join(wordlists.words(sample_id))
    )
    
    # print(cmd)
    cp = subprocess.run(cmd, shell=True, check=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    # print('------')
    # print(cp.stdout.decode('utf-8'))

    tagged_words = []
    for line in cp.stdout.decode('utf-8').split('\n'):
        if line.strip() == '':
            continue
        
        parts = line.split('\t')
        if len(parts) != 2:
            continue
        
        tagged_words.append(parts)

    print(tagged_words)
