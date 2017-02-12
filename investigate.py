#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 14:26:28 2017

@author: eightbit
"""

# avoid decoding problems
import sys
import os  
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import load_data

df = load_data()

##############################################################################
# Question lengths
##############################################################################
questions_df = pd.DataFrame()
questions_df['question'] = pd.concat([df['question1'], df['question2']], ignore_index=True)
questions_df['length'] = questions_df['question'].apply(len)
print "Max question length:", questions_df['length'].max()
print "Min question length:", questions_df['length'].min()
print "AVG question length:", questions_df['length'].mean()
print "STD question length:", questions_df['length'].std()

print questions_df.loc[questions_df['length']<10]

idxs = questions_df.length.argsort()
questions_df.iloc[idxs[0:250]]
