import numpy as np
import os
import csv
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import random
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import csr_matrix, csc_matrix
x = np.array([[1,0,0], [0,2,0], [1,1,0]])
test_row = csr_matrix(x)
test_col = csc_matrix(x)
non_zero = np.nonzero(x)
non_zero_ = np.nonzero(test_row)

# This file replicates the work of Carvalho et al.
# They used uni-grams OR bi-grams as features. And linear SVM as classification. Applied only 2-class classification,
filename = '../LSTM-CNN code/OnlyFirstSet_TrainingSet_2Class.csv'
with open(filename, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    training_tweets = []
    for tweet in reader:
        tweet[2] = tweet[2].lower()
        training_tweets.append(tweet)

# Data preparation: 1.Lower case, 2. Remove stop words. 3, remove punctuation mark.

tweet_corpus = [tweet[2] for tweet in training_tweets]
# standard bag-of-words representation
unigram = (1, 1)
bigram = (2, 2)
vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=unigram, analyzer=u'word')
Train_X = vectorizer.fit_transform(tweet_corpus)
transformer = TfidfTransformer(smooth_idf=False)
# Train_X = transformer.fit_transform(Train_X)
training_keywords = vectorizer.get_feature_names()

# Convert bag of words to binary values. We use binary features since nothing has said about this.
Train_X[np.nonzero(Train_X)] = 1
# Traing Y and Labels
Train_Y = [int(tweet[0]) for tweet in training_tweets]

# Do the same as training tweets for test tweet.
filename = '../LSTM-CNN code/OnlyFirstSet_TestSet_2Class.csv'
with open(filename, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    test_tweets = []
    for tweet in reader:
        tweet[2] = tweet[2].lower()
        test_tweets.append(tweet)

tweet_corpus = [tweet[2] for tweet in test_tweets]
# standard bag-of-words representation

vectorizer = CountVectorizer(min_df=1, ngram_range=unigram, analyzer=u'word', vocabulary=training_keywords)
Test_X = vectorizer.fit_transform(tweet_corpus)

# Convert bag of words to binary values.
Test_X[np.nonzero(Test_X)] = 1
# Traing Y and Labels
Test_Y = [int(tweet[0]) for tweet in test_tweets]

with open('OnlyFirstSet_Carvalho_2class.pickle', 'wb') as f:
    pickle.dump([Train_X, Test_X, Train_Y, Test_Y], f)

