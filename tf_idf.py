from terms_dictionary import DictionaryTrie
from options import *
from os.path import join
import math
import numpy as np
import csv

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
full_corpus = DictionaryTrie()
categories_dictionary = {}
for category in CATEGORIES:
    categories_dictionary[category]=DictionaryTrie.build_category_dictionary(join(CATEGORY_PATH, category))
    full_corpus.extend(categories_dictionary[category])

# Find stop words
stop_words = full_corpus.find_n_max_elements(STOP_WORDS_COUNTER)
stop_words.sort()
stop_words.reverse()
print("List of stop words:", stop_words)


# Evaluate TF-IDF for every category
for category, trie in categories_dictionary.items():
    print("Evaluate TF-IDF for category {}".format(category))
    for word, frequency in trie.iteritems():
        word_vector = [1 if trie.get(word) else 0 for trie in categories_dictionary.values()]
        value = calc_tf_idf(frequency, trie.size, word_vector)
        trie[word] = value

# Delete stop words from categories' tries
for category, trie in categories_dictionary.items():
    for frequency, word in stop_words:
        if trie.get(word):
            del trie[word]

# Build terms dictionary
terms_dictionary = {}
for category, trie in categories_dictionary.items():
    if category not in TERM_CATEGORIES:
        continue
    terms_dictionary[category] = trie.find_n_max_elements(TERMS_COUNTER)
    terms_dictionary[category].sort()
    terms_dictionary[category].reverse()
    print("List of terms for category {}:".format(category), terms_dictionary[category])

# Save results as csv
print("Save results as csv")
with open('data/results/stop_words.csv','w') as out:
    stop_words_out=csv.writer(out)
    stop_words_out.writerow(['stop word', 'frequency'])
    for frequency, word in stop_words:
        stop_words_out.writerow([word, frequency])

with open('data/results/terms.csv','w') as out:
    terms_out=csv.writer(out)
    terms_out.writerow(['category','term', 'tf-idf'])
    for category, terms in terms_dictionary.items():
        for value, term in terms:
            terms_out.writerow([category, term, value])




