import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
import glob
import os
import string
from nltk.corpus import stopwords


###############################################make a new folder if it doesnot exists################################
newpath = 'output_data' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
#########################accessing files from the folder ########################################
path = '/home/aziya/pythonpractice/test_data/*.txt' 
files = glob.glob(path)
stop_words = set(stopwords.words('urdu'))
#########################################preprocessing###########################3

def processing():
    for file in files:  
        print(file)
        f=open(file, 'r')
        content = f.read() 
        print(content)
        for sent in sent_tokenize(str(content)):                                                     #tokenization
            words = word_tokenize(sent)
            translator = str.maketrans({key: None for key in string.punctuation})  #punctuation removal
            words = [s.lower().translate(translator) for s in words]
        for w in words:              #stopword removal
                if w not in stop_words:
                    output = os.path.basename(file) #get name of file
                    open('output_data/' + output, 'w').write(str(w)) #save output
processing()
