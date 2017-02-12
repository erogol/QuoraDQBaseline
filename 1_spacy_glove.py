# -*- coding: utf-8 -*-
'''
    It is really simple algorithm based on word2vec
    - convert mean word to vec representations of the questions
    - train a simple model for pairs and see the difference
'''
# avoid decoding problems
import sys
import os  
#reload(sys)  
#sys.setdefaultencoding('utf8')

import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv("/media/eightbit/8bit_5tb/NLP_data/Quora/DuplicateQuestion/quora_duplicate_questions.tsv",delimiter='\t')

# encode questions to unicode
df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))

if os.path.exists('data/1_df.pkl'):
    df = pd.read_pickle('data/1_df.pkl')
else:
    # exctract word2vec vectors
    import spacy
    nlp = spacy.load('en')
    
    vecs1 = [doc.vector for doc in nlp.pipe(df['question1'], n_threads=50)]
    vecs1 =  np.array(vecs1)
    df['q1_feats'] = list(vecs1)
    
    vecs2 = [doc.vector for doc in nlp.pipe(df['question2'], n_threads=50)]
    vecs2 =  np.array(vecs2)
    df['q2_feats'] = list(vecs2)

    # save features
    pd.to_pickle(df, 'data/1_df.pkl')
    
from scipy.spatial.distance import euclidean
vec1 = df[df['qid1']==97]['q1_feats'].values
vec2 = df[df['qid2']==98]['q2_feats'].values
dist = euclidean(vec1[0], vec2[0])
print("dist btw duplicate: %f" % (dist))


vec1 = df[df['qid1']==91]['q1_feats'].values
vec2 = df[df['qid2']==92]['q2_feats'].values
dist = euclidean(vec1[0], vec2[0])
print("dist btw non-duplicate: %f" % (dist))

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
#mean =X_train.mean(axis=2)
#std =X_train.std(axis=2)
#X_train_norm = (X_train - mean[:,:,None])/(std[:,:,None]+1e-8)
#
#mean =X_test.mean(axis=2)
#std =X_test.std(axis=2)
#X_test_norm = (X_test - mean[:,:,None])/(std[:,:,None]+1e-8)

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
##############################################################################           
# create model
from siamese import *
from keras.optimizers import RMSprop, SGD
net = create_network(300)

# train
optimizer = SGD(lr=0.1, momentum=0.8, nesterov=True, decay=0.004)
#optimizer = RMSprop(lr=0.001)
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
