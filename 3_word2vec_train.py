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

# encode questions to unicode
df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))

##############################################################################
# TRAIN GLOVE
##############################################################################
import gensim

if os.path.exists('data/3_word2vec.mdl'):
    model = gensim.models.Word2Vec.load_word2vec_format('data/3_word2vec.bin', binary=True)
    # trim memory
    model.init_sims(replace=True)
    # creta a dict 
    w2v = dict(zip(model.index2word, model.syn0))
    print "Number of tokens in Word2Vec:", len(w2v.keys())
else:
    questions = list(df['question1']) + list(df['question2'])
    
    # tokenize
    c = 0
    for question in tqdm(questions):
        questions[c] = list(gensim.utils.tokenize(question, deacc=True, lower=True))
        c += 1
    
    # train model
    model = gensim.models.Word2Vec(questions, size=300, workers=16, iter=10, negative=20)
    
    # trim memory
    model.init_sims(replace=True)
    
    # creta a dict 
    w2v = dict(zip(model.index2word, model.syn0))
    print "Number of tokens in Word2Vec:", len(w2v.keys())
    
    # save model
    model.save('data/3_word2vec.mdl')
    model.save_word2vec_format('data/3_word2vec.bin', binary=True)
    del questions
    
##############################################################################
# EXTRACT FEATURES
##############################################################################
from utils import MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer

if os.path.exists('data/3_df.pkl'):
    df = pd.read_pickle('data/3_df.pkl')
else:
    # gather all questions
    questions = list(df['question1']) + list(df['question2'])
    
    # tokenize questions
    c = 0
    for question in tqdm(questions):
        questions[c] = list(gensim.utils.tokenize(question, deacc=True))
        c += 1

#    me = MeanEmbeddingVectorizer(w2v)
    me = TfidfEmbeddingVectorizer(w2v)
    me.fit(questions)
    # exctract word2vec vectors
    vecs1 = me.transform(df['question1'])
    df['q1_feats'] = list(vecs1)
    
    vecs2 = me.transform(df['question2'])
    df['q2_feats'] = list(vecs2)
    
    # save features
    pd.to_pickle(df, 'data/3_df.pkl')

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

del b
del q1_feats
del q2_feats

# preprocess data, mean center unit std
#from sklearn.preprocessing import normalize
#X_train_norm = np.zeros_like(X_train)
#X_train_norm[:,0,:] = normalize(X_train[:,0,:], axis=0)
#X_train_norm[:,1,:] = normalize(X_train[:,1,:], axis=0)
#X_test_norm = np.zeros_like(X_test)
#X_test_norm[:,0,:] = normalize(X_test[:,0,:], axis=0)
#X_test_norm[:,1,:] = normalize(X_test[:,1,:], axis=0)

##############################################################################
# TRAIN MODEL
# 3 layers resnet (before relu) + adam + layer concat : 0.68
# 3 layers resnet (before relu) + adam + layer concat + 20 negative sampling: ?
##############################################################################           
# create model
from siamese import *
from keras.optimizers import RMSprop, SGD, Adam
net = create_network(300)

# train
#optimizer = SGD(lr=0.01, momentum=0.8, nesterov=True, decay=0.004)
optimizer = Adam(lr=0.001)
net.compile(loss=contrastive_loss, optimizer=optimizer)

for epoch in range(50):
    net.fit([X_train[:,0,:], X_train[:,1,:]], Y_train,
          validation_data=([X_test[:,0,:], X_test[:,1,:]], Y_test),
          batch_size=128, nb_epoch=1, shuffle=True)
    
    # compute final accuracy on training and test sets
    pred = net.predict([X_test[:,0,:], X_test[:,1,:]])
    te_acc = compute_accuracy(pred, Y_test)
    
#    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
