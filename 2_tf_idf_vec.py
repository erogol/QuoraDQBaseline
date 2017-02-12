# -*- coding: utf-8 -*-
'''
    It is really simple algorithm based on word2vec
    - convert mean word to vec representations of the questions
    - train a simple model for pairs and see the difference
'''
# avoid decoding problems
import sys
import os  
import pandas as pd
import numpy as np
from tqdm import tqdm
##############################################################################
# LOAD DATA
##############################################################################

df = pd.read_csv("/media/eightbit/8bit_5tb/NLP_data/Quora/DuplicateQuestion/quora_duplicate_questions.tsv",delimiter='\t')

# print simple stats
print("number of rows (question pairs): %i"%(df.shape[0]))
print(df['is_duplicate'].value_counts())

# find unique question ids
unique_qids = set(list(df['qid2'].unique()) + list(df['qid1'].unique()))
print("number of unique questions: %i" % (len(unique_qids)))

# encode questions to unicode
df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))

##############################################################################
# TFIDF
##############################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# merge texts
questions = list(df['question1']) + list(df['question2'])

tfidf = TfidfVectorizer(lowercase=False, )
tfidf.fit_transform(questions)

# dict key:word and value:tf-idf score
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
del questions
##############################################################################
# WORD2VEC
##############################################################################
if os.path.exists('data/2_word2vec_tfidf.pkl'):
    df = pd.read_pickle('data/2_word2vec_tfidf.pkl')
else:
    # exctract word2vec vectors
    import spacy
    nlp = spacy.load('en')
    
    vecs1 = []
    for qu in tqdm(list(df['question1'])):
        doc = nlp(qu) 
        mean_vec = np.zeros([len(doc), 300])
        for word in doc:
            # word2vec
            vec = word.vector
            # fetch df score
            try:
                idf = word2tfidf[str(word)]
            except:
                #print word
                idf = 0
            # compute final vec
            mean_vec += vec * idf
        mean_vec = mean_vec.mean(axis=0)
        vecs1.append(mean_vec)
    df['q1_feats'] = list(vecs1)
    
    vecs2 = []
    for qu in tqdm(list(df['question2'])):
        doc = nlp(qu) 
        mean_vec = np.zeros([len(doc), 300])
        for word in doc:
            # word2vec
            vec = word.vector
            # fetch df score
            try:
                idf = word2tfidf[str(word)]
            except:
                print word
                idf = 0
            # compute final vec
            mean_vec += vec * idf
        mean_vec = mean_vec.mean(axis=0)
        vecs2.append(mean_vec)
    df['q2_feats'] = list(vecs2)

    # save features
    pd.to_pickle(df, 'data/2_word2vec_tfidf.pkl')

##############################################################################
# CREATE TRAIN DATA
##############################################################################
# shuffle df
df = df.reindex(np.random.permutation(df.index))

# set number of train and test instances
num_train = int(df.shape[0] * 0.88)
num_test = df.shape[0] - num_train                 
print("Number of training pairs: %i"%(num_train))
print("Number of testing pairs: %i"%(num_test))

# init data data arrays
X_train = np.zeros([num_train, 2, 300])
X_test  = np.zeros([num_test, 2, 300])
Y_train = np.zeros([num_train]) 
Y_test = np.zeros([num_test])

# format data 
b = [a[None,:] for a in list(df['q1_feats'].values)]
q1_feats = np.concatenate(b, axis=0)

b = [a[None,:] for a in list(df['q2_feats'].values)]
q2_feats = np.concatenate(b, axis=0)

# fill data arrays with features
X_train[:,0,:] = q1_feats[:num_train]
X_train[:,1,:] = q2_feats[:num_train]
Y_train = df[:num_train]['is_duplicate'].values
            
X_test[:,0,:] = q1_feats[num_train:]
X_test[:,1,:] = q2_feats[num_train:]
Y_test = df[num_train:]['is_duplicate'].values

# remove useless variables
del b
del q1_feats
del q2_feats

# preprocess data, unit std
#X_train_norm = np.zeros_like(X_train)
#d = (np.sum(X_train[:,0,:] ** 2, 1) ** (0.5))
#X_train_norm[:,0,:] = (X_train[:,0,:].T / (d + 1e-8)).T
#d = (np.sum(X_train[:,1,:] ** 2, 1) ** (0.5))
#X_train_norm[:,1,:] = (X_train[:,1,:].T / (d + 1e-8)).T
#            
#            
#X_test_norm = np.zeros_like(X_test)
#d = (np.sum(X_test[:,0,:] ** 2, 1) ** (0.5))
#X_test_norm[:,0,:] = (X_test[:,0,:].T / (d + 1e-8)).T
#d = (np.sum(X_test[:,1,:] ** 2, 1) ** (0.5))
#X_test_norm[:,1,:] = (X_test[:,1,:].T / (d + 1e-8)).T

##############################################################################
# TRAIN MODEL
# - 2 layers net : 0.67
# - 3 layers net + adam : 0.74
# - 3 layers resnet (after relu) + adam : 0.78
# - 3 layers resnet (before relu) + adam : 0.77
# - 3 layers resnet (before relu) + adam + dropout : 0.75
# - 3 layers resnet (before relu) + adam + layer concat : 0.79
# - 3 layers resnet (before relu) + adam + layer concat + unit_norm : 0.77
# - 3 layers resnet (before relu) + adam + unit_norm + cosine_distance : Fail
############################################################################## 
          
# create model
from siamese import *
from keras.optimizers import RMSprop, SGD, Adam
net = create_network(300)

# train
#optimizer = SGD(lr=1, momentum=0.8, nesterov=True, decay=0.004)
optimizer = Adam(lr=0.001)
net.compile(loss=contrastive_loss, optimizer=optimizer)

for epoch in range(50):
    net.fit([X_train_norm[:,0,:], X_train_norm[:,1,:]], Y_train,
          validation_data=([X_test_norm[:,0,:], X_test_norm[:,1,:]], Y_test),
          batch_size=128, nb_epoch=1, shuffle=True, )
    
    # compute final accuracy on training and test sets
    pred = net.predict([X_test_norm[:,0,:], X_test_norm[:,1,:]], batch_size=128)
    te_acc = compute_accuracy(pred, Y_test)
    
#    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
