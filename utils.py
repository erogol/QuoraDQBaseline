import argparse
import sys
from gensim.utils import tokenize
import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import pandas as pd

###############################################################################
# Merginf TF-IDF scores with Word2Vec
###############################################################################
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in list(tokenize(words , deacc=True)) if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
###############################################################################
# Stats on WordVec vectors
##############################################################################
def compare_two_pairs(pair_num, data):
    a = data[pair_num,0,:]
    b = data[pair_num,1,:]
    return euclidean(a,b)

def compute_feats_discrimination(data, labels):
    unique_labels = np.unique(labels)
    avg_dist_dict = {}
    for label in unique_labels:
        sub_data = data[labels==label]
        sub_res_data = data[labels!=label]
        dists = []
        for i in tqdm(range(sub_data.shape[0])):
            # compute pari distance
            pair_dist = compare_two_pairs(i, sub_data)
            
            # compute mean non pair distance
            idxs = np.random.permutation(sub_res_data.shape[0])[0:10]
            non_pair_dist = cdist(sub_res_data[idxs,0,:], sub_data[i,1,:][None,:], metric='euclidean').mean()
            
            # append to results
            dists.append([dist, non_pair_dist])
        avg_dist = np.mean(dists, axis=1)
        avg_dist_dict[label] = avg_dist
    return avg_dist_dict

###############################################################################
# Evaluation functions for Stanford GLOVE vectors
##############################################################################
def generate_glove(vocab_file, vectors_file):
    with open(vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)


def distance_glove(W, vocab, ivocab, input_term):
    for idx, term in enumerate(input_term.split(' ')):
        if term in vocab:
            print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term], :])
            else:
                vec_result += W[vocab[term], :] 
        else:
            print('Word: %s  Out of dictionary!\n' % term)
            return
    
    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(W, vec_norm.T)

    for term in input_term.split(' '):
        index = vocab[term]
        dist[index] = -np.Inf

    a = np.argsort(-dist)[:N]

    print("\n                               Word       Cosine distance\n")
    print("---------------------------------------------------------\n")
    for x in a:
        print("%35s\t\t%f\n" % (ivocab[x], dist[x]))


###############################################################################
# Evaluation functions for Stanford GLOVE vectors
###############################################################################

def load_data():
    df = pd.read_csv("/media/eightbit/8bit_5tb/NLP_data/Quora/DuplicateQuestion/quora_duplicate_questions.tsv",delimiter='\t')

    # encode questions to unicode
    df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
    df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))

    return df

    

 