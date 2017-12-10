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
from gensim.models.keyedvectors import KeyedVectors
from scipy.sparse import csr_matrix

x = np.array([[1,0,0], [0,2,0], [1,1,0]])
g = csr_matrix(x)
c = csr_matrix(x)
b = np.vstack((c, g))
x[np.nonzero(x)] = 2
os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\LSTM-CNN code')

# This file replicates the work of Periera et al.
# They combined bag-of-words and word embeddings as vector representation. Then, apply the classification
# They classify tweets into two groups: travel-related tweets vs non-related. This is the only task in their paper.
filename = '../LSTM-CNN code/1_TrainingSet_3Class.csv'
with open(filename, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    training_tweets = []
    for tweet in reader:
        tweet[2] = tweet[2].lower()
        training_tweets.append(tweet)

# Data preparation: 1.Lower case, 2. trasnfrom repeating characters, 3. Remover URL and user mention.
# 2 and 3 have been already applied in the data. 1 is applied in the above data parsing.

# Tokenizing and removing stopping words by sklearn vectorizer.
# stop words are words occurring in more than 60% documents according to the paper.
vectorizer = CountVectorizer(min_df=1, stop_words=None, max_df=0.60, ngram_range=(1, 1), analyzer=u'word')
tweet_corpus = [tweet[2] for tweet in training_tweets]
DocumentTerm = vectorizer.fit_transform(tweet_corpus)
DocumentTerm_Array = DocumentTerm.toarray()
Name = vectorizer.get_feature_names()
Frequency = np.sum(DocumentTerm_Array, axis=0)
Sort_Index = np.argsort(Frequency)
Sort_Index = Sort_Index[::-1]
Sort_Index = Sort_Index[:3000]   # based on the paper, the first 3000 frequent words retrieved.
keywords = [Name[i] for i in Sort_Index]

# standard bag-of-words representation
vectorizer = CountVectorizer(min_df=1, stop_words=None, ngram_range=(1, 1), analyzer=u'word', vocabulary=keywords)
bagofwords_training = vectorizer.fit_transform(tweet_corpus).toarray()
# Convert bag of words to binary values.
bagofwords_training[np.nonzero(bagofwords_training)] = 1

# word embeddings.
vectorizer = CountVectorizer(min_df=1, stop_words=None, ngram_range=(1, 1), analyzer=u'word')
Word2Vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
analyze = vectorizer.build_analyzer()
tweet_embeddings_training = np.zeros((len(tweet_corpus), Word2Vec_model.vector_size))
for i in range(len(tweet_corpus)):
    tweet_words = analyze(tweet_corpus[i])
    word2vec = []
    for word in tweet_words:
        try:
            word2vec.append(Word2Vec_model[word])
        except KeyError:
            pass
    if word2vec == []:
        continue
    tweet_embeddings_training[i, :] = np.mean(np.array(word2vec), axis=0)

# concatenate bag-of-words and word embeddings.
Train_X = np.hstack((tweet_embeddings_training, bagofwords_training))
Train_X = csr_matrix(Train_X)
Train_Y = [int(tweet[0]) for tweet in training_tweets]

# Create feature representation for test set in the same process as training.
# According to the paper, the 3000 frequent words in training set is used for test as well.
filename = '../LSTM-CNN code/1_TestSet_3Class.csv'
with open(filename, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    test_tweets = []
    for tweet in reader:
        tweet[2] = tweet[2].lower()
        test_tweets.append(tweet)

tweet_corpus = [tweet[2] for tweet in test_tweets]

# standard bag-of-words representation
vectorizer = CountVectorizer(min_df=1, stop_words=None, ngram_range=(1, 1), analyzer=u'word', vocabulary=keywords)
bagofwords_test = vectorizer.fit_transform(tweet_corpus).toarray()
# Convert bag of words to binary values.
bagofwords_test[np.nonzero(bagofwords_test)] = 1

# word embeddings.
vectorizer = CountVectorizer(min_df=1, stop_words=None, ngram_range=(1, 1), analyzer=u'word')
analyze = vectorizer.build_analyzer()
tweet_embeddings_test = np.zeros((len(tweet_corpus), Word2Vec_model.vector_size))
for i in range(len(tweet_corpus)):
    tweet_words = analyze(tweet_corpus[i])
    word2vec = []
    for word in tweet_words:
        try:
            word2vec.append(Word2Vec_model[word])
        except KeyError:
            pass
    if word2vec == []:
        continue
    tweet_embeddings_test[i, :] = np.mean(np.array(word2vec), axis=0)

# concatenate bag-of-words and word embeddings.
Test_X = np.hstack((tweet_embeddings_test, bagofwords_test))
Test_X = csr_matrix(Test_X)
Test_Y = [int(tweet[0]) for tweet in test_tweets]

with open('1-Periera_3class.pickle', 'wb') as f:
    pickle.dump([Train_X, Test_X, Train_Y, Test_Y], f)

