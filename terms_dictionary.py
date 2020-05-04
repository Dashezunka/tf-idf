import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pygtrie
from heapq import heapify, heappush, heappop
from os import listdir
from os.path import isfile, join

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def tokenize_text(text):
    text = re.sub(r"$\d+\W+|\b\d+\b|\W+\d+$", " ", text)
    text = re.sub(r'[^\w\s]', '', text)
    return word_tokenize(text.lower())

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = list()
    for token, pos in pos_tag(tokens):
        wordnet_pos = get_wordnet_pos(pos)
        if wordnet_pos=='':
            lemmas.append(token)
        else:
            lemmas.append(lemmatizer.lemmatize(token, get_wordnet_pos(pos)))
    return lemmas

class DictionaryTrie:
    def __init__(self):
        self.trie = pygtrie.CharTrie()
        self.size = 0

    def find_n_max_elements(self, n):
        result_heap = []
        heapify(result_heap)
        for key, value in self.trie.iteritems():
            if len(result_heap) < n:
                heappush(result_heap, (value, key))
            else:
                head = heappop(result_heap)
                if value >= head[0]:
                    heappush(result_heap, (value, key))
                else:
                    heappush(result_heap, head)
        return result_heap

    def get_items_more_than_threshold(self, threshold):
        items = []
        for key, value in self.trie.iteritems():
            if value > threshold:
                items.append((key, value))
        return items

    def get_items_less_than_threshold(self, threshold):
        items = []
        for key, value in self.trie.iteritems():
            if value < threshold:
                items.append((key, value))
        return items

    def extend(self, another_trie):
        for key,value in another_trie.iteritems():
            self.trie[key] = self.trie.get(key, 0) + value
            self.size += value

    def __getitem__(self, item):
        return self.trie[item]

    def __setitem__(self, key, value):
        self.trie[key] = value

    def iteritems(self):
        return self.trie.iteritems()

    def add_words(self, words):
        for w in words:
            self.trie[w] = self.trie.get(w, 0) + 1
        self.size += len(words)

    def get(self, key, default=None):
        return self.trie.get(key, default)

    def __delitem__(self, key_or_slice):
        del self.trie[key_or_slice]



    @staticmethod
    def build_document_dictionary(doc_path):
        corpus = DictionaryTrie()
        with open(doc_path, 'r') as doc:
            text = doc.read()
            tokens = tokenize_text(text)
            lemmas = lemmatize_text(tokens)
            corpus.add_words(lemmas)
        return corpus

    @staticmethod
    def build_category_dictionary(category_path):
        document_list = [f for f in listdir(category_path) if isfile(join(category_path, f))]
        corpus = DictionaryTrie()
        for document in document_list:
            with open(join(category_path, document), 'r') as doc:
                text = doc.read()
                tokens = tokenize_text(text)
                lemmas = lemmatize_text(tokens)
                corpus.add_words(lemmas)
        return corpus

