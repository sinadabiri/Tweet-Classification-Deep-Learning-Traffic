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
from itertools import combinations
from nltk.corpus import wordnet as wn

a = [1, 2, 3, 4]
b = 'Huddled on the same block on Avenue Rd., florists have created a tourist attraction: Toronto flower district.'
g = 'huddled' in b.lower()

os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\LSTM-CNN code')

# This file replicate the work of Gu et al.
# Their main module is the adaptive data acquisition, which pulls out the most relevant traffic-incident tweets
'''
# Appling the GU method. It has been commented out since the final keywords have already been obtained. 
# Starting adaptive data acquisition and identifying keywords for the classification task. 
with open('TrainingSet_2Class.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    training_tweets = []
    for tweet in reader:
        training_tweets.append(tweet)

# Create an initial dictionary with 50 keywords (I choose it from my own randomly)

with open('All-TI-Keywords-Weights.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    # I initially add whatever keywords mentioned in their paper, remaining choosing from my own
    keywords = []
    words_paper = ['traffic', 'accident', 'road', 'avenue', 'car', 'bike', 'truck', 'driver', 'injured', 'congestion',
                   'slow', 'exit', 'mile', 'stop']
    for row in reader:
        if row[0] not in words_paper:
            keywords.append(row[0])
random.shuffle(keywords)

# Initial keywords. 50 as said in paper.
sv = words_paper + keywords
sv = sv[:50]

def expansion_operator(sv):
    expand_sv = sv.copy()
    for word in sv:
        try:
            syn = wn.synsets(word)[0].lemma_names()
            if word in syn:
                syn.remove(word)
            [expand_sv.append(word) for word in syn[:2]]
        except IndexError:
            pass
    return expand_sv

#sv = expansion_operator(sv)
#current_training = training_tweets[: 4000]
def acquired_tweets(expand_sv, current_training):
    """
    Acquired tweets from one, two, three, and four words combination. Words selected from dict (sv).
    :param
    :return: acquired Tweets
    """
    Wv = []
    for tweet in current_training:
        for query in expand_sv:
            if any(word in tweet[1].lower() for word in query):
                Wv.append(tweet)
                break
    return Wv

# Wv = acquired_tweets(sv)
# Put ngram as 5 since at the end we want to get the maximum counts.
# It is almost impossible to have more than 5-word token combination in the final list.
# In this step, for each tweet, I only count the token combinations that occure in sequence not all possible token comb.
vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 5), analyzer=u'word')
def count_tokens(Wv, sv):
    tweets_traffic = [tweet[1] for tweet in Wv if tweet[0] == '1']
    DocumentTerm_Array = vectorizer.fit_transform(tweets_traffic).toarray()
    Name = vectorizer.get_feature_names()
    Frequency = np.sum(DocumentTerm_Array, axis=0)
    Sort_Index = np.argsort(Frequency)
    Sort_Index = Sort_Index[::-1]
    positive_words = []
    counter = 0
    for i in range(len(Sort_Index)):
        if Name[Sort_Index[i]] not in sv:
            positive_words.append(Name[Sort_Index[i]])
            counter += 1
        # in each iteration (total is 9), we add only 9 words to dict, which is the same as paper.
        if counter >= 9:
            break
    return positive_words + sv

# we have 50 initial words, we need 81 more (to be the same as total words in the paper 131)
# we have almost 41000 training set. In each of 9 iteration, we apply the method in 4550 tweet to get 9
# positive words to Traffic
counter = 0
for i in range(9):
    expand_sv = expansion_operator(sv)
    current_training = training_tweets[counter:counter + 4550]
    Wv = acquired_tweets(expand_sv, current_training)
    new_sv = count_tokens(Wv, sv)
    sv = new_sv.copy()
    counter += 4550

with open('keywords_Gu_Study.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(sv, f)
'''
# Now we obtain the keywords. Then, apply standard bag of word representation, where n-gram is tha max no. of words
# in all keywords.
with open('keywords_Gu_Study.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    sv = pickle.load(f)
filename = '../LSTM-CNN code/1_TrainingSet_3Class.csv'
with open(filename, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    training_tweets = []
    for tweet in reader:
        tweet[2] = tweet[2].lower()
        training_tweets.append(tweet)

# determine n in n-grams.
n = max([len(wordset.split()) for wordset in sv])
tweet_corpus = [tweet[2] for tweet in training_tweets]
# standard bag-of-words representation
vectorizer = CountVectorizer(min_df=1, stop_words=None, ngram_range=(1, n), analyzer=u'word', vocabulary=sv)
Train_X = vectorizer.fit_transform(tweet_corpus)
# Convert bag of words to binary values. As said in paper.
Train_X[np.nonzero(Train_X)] = 1
# Traing Y and Labels
Train_Y = [int(tweet[0]) for tweet in training_tweets]

# Do the same as training tweets for test tweet.
filename = '../LSTM-CNN code/1_TestSet_3Class.csv'
with open(filename, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    test_tweets = []
    for tweet in reader:
        tweet[2] = tweet[2].lower()
        test_tweets.append(tweet)

tweet_corpus = [tweet[2] for tweet in test_tweets]
# standard bag-of-words representation
vectorizer = CountVectorizer(min_df=1, ngram_range=(1, n), analyzer=u'word', vocabulary=sv)
Test_X = vectorizer.fit_transform(tweet_corpus)

# Convert bag of words to binary values.
Test_X[np.nonzero(Test_X)] = 1
# Traing Y and Labels
Test_Y = [int(tweet[0]) for tweet in test_tweets]

with open('1-Gu_3class.pickle', 'wb') as f:
    pickle.dump([Train_X, Test_X, Train_Y, Test_Y], f)