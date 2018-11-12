import os
import sys
import shutil

from corpus_processor import get_lda_input_from_corpus_folder

from lda_model import LDAModel


def setup_temp_corpus_folder(filename):
    os.system('rm -rf ' + LDAModel.TEMP_CORPUS_FOLDER + '/*')
    shutil.copy(filename, LDAModel.TEMP_CORPUS_FOLDER)


if __name__ == '__main__':

    if 3 != len(sys.argv):
        print("\nUsage: %s category_name\n" % sys.argv[0])
        sys.exit(1)

    # ========================================
    obj = LDAModel()

    lda_input = get_lda_input_from_corpus_folder(LDAModel.TEMP_CORPUS_FOLDER)

    dic, corp, mod =obj.lda_test(lda_input)
    topics = mod.print_topics(num_words=7)
    for topic in topics:
        print(topic)
