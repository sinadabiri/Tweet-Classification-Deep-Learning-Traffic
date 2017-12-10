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

# This file replicate the work of Andera et al.
# The first applied stop-word and word stemming as the preprocessing techniques.
# Then, they applied feature selection (Information gain based on Gini).
# They used SVM to classify the features.

filename = '../LSTM-CNN code/1_TrainingSet_3Class.csv'
with open(filename, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    training_tweets = []
    for tweet in reader:
        tweet[2] = tweet[2].lower()
        training_tweets.append(tweet)

# Tokenizing and removing stopping words by sklearn vectorizer.
vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
analyze = vectorizer.build_analyzer()

# Stemming the words in each tweet using NLTK function (Porter Stemmer)
stemmer = PorterStemmer()
stemmed_tweets = []
tweet_corpus = [tweet[2] for tweet in training_tweets]
for tweet in tweet_corpus:
    tweet = analyze(tweet)
    tweet = [stemmer.stem(word) for word in tweet]
    stemmed_tweets.append(tweet)

# Collect all stems in all tweets in the collection
all_stems = [stem for sublist in stemmed_tweets for stem in sublist]
# Collect the unique stems
unique_stems = list(set(all_stems))

# Vectorizing (tokenizing) the original tweets using the vocabulary of unique stems.
vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), analyzer=u'word', vocabulary=unique_stems)
# Convert the tokenized vectors to bag of word representation with only term frequency.
features_counts = vectorizer.fit_transform(tweet_corpus)
Name = vectorizer.get_feature_names()
# Convert bag of words with term frequency to tf-idf as new weights.
transformer = TfidfTransformer(smooth_idf=True)
features_tfidf = transformer.fit_transform(features_counts)

# Feature selection process based on a tree model.
Train_Y = [int(tweet[0]) for tweet in training_tweets]

clf = DecisionTreeClassifier()
clf = clf.fit(features_tfidf, Train_Y)
feature_importance = clf.feature_importances_
# We set threshold as 1*mean to be almost the same as paper that chooses positive information gain value
model = SelectFromModel(clf, prefit=True, threshold="mean")  # threshold="3*mean", or"1.25median"
Train_X = model.transform(features_tfidf)
final_keywords = [Name[i] for i, item in enumerate(model.get_support()) if item==True]


# Do the same for test set
filename = '../LSTM-CNN code/1_TestSet_3Class.csv'
with open(filename, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    test_tweets = []
    for tweet in reader:
        tweet[2] = tweet[2].lower()
        test_tweets.append(tweet)

# Tokenizing and removing stopping words by sklearn vectorizer.
vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), analyzer=u'word', stop_words='english')
analyze = vectorizer.build_analyzer()
# Stemming the words in each tweet using NLTK function (Porter Stemmer)
stemmer = PorterStemmer()
stemmed_tweets = []
tweet_corpus = [tweet[2] for tweet in test_tweets]
for tweet in tweet_corpus:
    tweet = analyze(tweet)
    tweet = [stemmer.stem(word) for word in tweet]
    stemmed_tweets.append(' '.join(tweet))

# Vectorizing (tokenizing) the original tweets using the vocabulary of unique stems.
vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), analyzer=u'word', vocabulary=final_keywords)
# Convert the tokenized vectors to bag of word representation with only term frequency.
features_counts = vectorizer.fit_transform(stemmed_tweets)
Name = vectorizer.get_feature_names()
# Convert bag of words with term frequency to tf-idf as new weights.
transformer = TfidfTransformer(smooth_idf=False)
Test_X = transformer.fit_transform(features_counts)
# Feature selection process based on a tree model.
Test_Y = [int(tweet[0]) for tweet in test_tweets]


with open('1-Andrea_3class.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([Train_X, Test_X, Train_Y, Test_Y], f)


