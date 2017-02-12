#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:26:44 2017

@author: eightbit
"""

# avoid decoding problems
import sys
import os  
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import generate_glove
from gensim.utils import tokenize
##############################################################################
# LOAD DATA
##############################################################################

df = pd.read_csv("/media/eightbit/8bit_5tb/NLP_data/Quora/DuplicateQuestion/quora_duplicate_questions.tsv",delimiter='\t')

# encode questions to unicode
df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))


##############################################################################
# CREATE WORD2VEC vectors
##############################################################################
if os.path.exists('data/4_df.pkl'):
    df = pd.read_pickle('data/4_df.pkl')
else:    
    import glove
#    gm = glove.Glove()
#    gm = gm.load_stanford('/media/eightbit/8bit_5tb/NLP_model/glove_models/glove.6B/glove.6B.300d.txt')
    
    feats = []
    not_found_token = []
    for question in tqdm(df['question1']):
        tokens = tokenize(question, lowercase=True, deacc=True)
        feat_vec = []
        for token in tokens:
            try:
                feat_vec.append(gm.word_vectors[gm.dictionary[token]])
            except:
                feat_vec.append(np.zeros([300]))
                not_found_token.append(token)
        feat_vec = np.mean(feat_vec,axis=0)
        feats.append(feat_vec)
    df['q1_feats'] = list(feats)

    feats = []
    for question in tqdm(df['question2']):
        tokens = tokenize(question, lowercase=True, deacc=True)
        feat_vec = []
        for token in tokens:
            try:
                feat_vec.append(gm.word_vectors[gm.dictionary[token]])
            except:
                feat_vec.append(np.zeros([300]))
                not_found_token.append(token)     
        feat_vec = np.mean(feat_vec,axis=0)
        feats.append(feat_vec)
    df['q2_feats'] = list(feats)

    # save features
    pd.to_pickle(df, 'data/4_df.pkl')
    
##############################################################################
# CREATE TRAIN DATA
##############################################################################
# drop NaN rows with useless questions
df = df.dropna()

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

del b
del q1_feats
del q2_feats

# preprocess data, unit std
X_train_norm = np.zeros_like(X_train)
d = (np.sum(X_train[:,0,:] ** 2, 1) ** (0.5))
X_train_norm[:,0,:] = (X_train[:,0,:].T / (d + 1e-8)).T
d = (np.sum(X_train[:,1,:] ** 2, 1) ** (0.5))
X_train_norm[:,1,:] = (X_train[:,1,:].T / (d + 1e-8)).T
            
            
X_test_norm = np.zeros_like(X_test)
d = (np.sum(X_test[:,0,:] ** 2, 1) ** (0.5))
X_test_norm[:,0,:] = (X_test[:,0,:].T / (d + 1e-8)).T
d = (np.sum(X_test[:,1,:] ** 2, 1) ** (0.5))
X_test_norm[:,1,:] = (X_test[:,1,:].T / (d + 1e-8)).T

##############################################################################
# TRAIN MODEL
# - 3 layers resnet (before relu) + adam : 0.77
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
