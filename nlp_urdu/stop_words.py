def remove_urdu_stopwords(urdu_stopwords, words):
    processed_word_list = []
    for word in words:
        if word not in urdu_stopwords:
            processed_word_list.append(word)

    return processed_word_list
