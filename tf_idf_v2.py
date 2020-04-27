from terms_dictionary import DictionaryTrie
from options import *
from os.path import join, isfile
from os import listdir
import math
import numpy as np
import csv
import copy

CATEGORIES = TERM_CATEGORIES + OTHER_CATEGORIES

def calc_tf(word_frequency, doc_length):
    return word_frequency/doc_length

def calc_idf(word_vector):
    return math.log10(len(word_vector) / np.sum(word_vector))

def calc_tf_idf(word_frequency, doc_length, word_vector):
    tf = calc_tf(word_frequency, doc_length)
    idf = calc_idf(word_vector)
    return tf * idf

# Make corpus trie for every category and full corpus trie for all categories
categories_dictionary = {}

all_documents_list = []
for category in CATEGORIES:
    category_path = join(CATEGORY_PATH, category)
    document_list = [f for f in listdir(category_path) if isfile(join(category_path, f))]
    category_dict_list = []
    for document in document_list:
        document_dict = DictionaryTrie.build_document_dictionary(join(category_path, document))
        category_dict_list.append(document_dict)
        all_documents_list.append(copy.deepcopy(document_dict))
    categories_dictionary[category] = category_dict_list

# Evaluate TF-IDF for all documents
for document_dict in all_documents_list:
    for word, frequency in document_dict.iteritems():
        word_vector = [1 if trie.get(word) else 0 for trie in all_documents_list]
        value = calc_tf_idf(frequency, document_dict.size, word_vector)
        document_dict[word] = value

# Find stop words
stop_words = set()
for document_dict in all_documents_list:
    words_with_low_tf_idf = document_dict.get_items_less_than_threshold(STOP_WORDS_THRESHOLD)
    for w, tf_idf in words_with_low_tf_idf:
        stop_words.add(w)
stop_words_list = list(stop_words)
stop_words_list.sort()
print("List of stop words:", stop_words_list)
print(len(stop_words_list))

# Make corpus of documents' dicts for every specific category
categories_corpus = {}
for category, dict_list in categories_dictionary.items():
    category_corpus = DictionaryTrie()
    for document_dict in dict_list:
        category_corpus.extend(document_dict)
    categories_corpus[category] = category_corpus


# Evaluate TF-IDF for every category
for category, category_corpus in categories_corpus.items():
    if category not in TERM_CATEGORIES:
        continue
    print("Evaluate TF-IDF for category {}".format(category))
    for word, frequency in category_corpus.iteritems():
        word_vector = [1 if category_corpus.get(word) else 0 for category_corpus in categories_corpus.values()]
        value = calc_tf_idf(frequency, category_corpus.size, word_vector)
        category_corpus[word] = value


# Delete stop words from categories' tries
for category, category_corpus in categories_corpus.items():
    if category not in TERM_CATEGORIES:
        continue
    for word in stop_words:
        if category_corpus.get(word):
            del category_corpus[word]

# Build terms dictionary
terms_dictionary = {}
for category, category_corpus in categories_corpus.items():
    if category not in TERM_CATEGORIES:
        continue
    terms_dictionary[category] = category_corpus.get_items_more_than_threshold(TERMS_THRESHOLD)
    terms_dictionary[category].sort(key = lambda x : x[1])
    terms_dictionary[category].reverse()
    print("List of terms for category {}:".format(category), terms_dictionary[category])

terms_to_remove = {}
for category, terms in terms_dictionary.items():
    if category not in TERM_CATEGORIES:
        continue
    terms_to_remove[category] = []
    print("Analyse terms for ", category)
    contrast_category_list = copy.deepcopy(TERM_CATEGORIES)
    contrast_category_list.remove(category)
    contrast_category = contrast_category_list[0]
    print("Contrast category is {}".format(contrast_category))
    for current_term, current_tf_idf in terms:
        contrast_tf_idf = categories_corpus[contrast_category].get(current_term)
        # print(current_term, current_tf_idf, contrast_tf_idf)
        if contrast_tf_idf and contrast_tf_idf > CONTRAST_THRESHOLD:
            terms_to_remove[category].append(current_term)
    for term in terms_to_remove[category]:
        category_dict = terms_dictionary[category]
        del category_dict[term]
    print("List of remaining terms for category {}:".format(category), terms_dictionary[category])

# Save results as csv
print("Save results as csv")
# for stop words
with open('data/results/stop_words_v2.csv','w') as out:
    stop_words_out=csv.writer(out)
    stop_words_out.writerow(['stop word'])
    for word in stop_words_list:
        stop_words_out.writerow([word])
# for terms
with open('data/results/terms_v2.csv','w') as out:
    terms_out=csv.writer(out)
    terms_out.writerow(['category','term', 'tf-idf'])
    for category, terms in terms_dictionary.items():
        for value, term in terms:
            terms_out.writerow([category, term, value])
