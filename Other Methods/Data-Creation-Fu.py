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

# This file replicate the work of Fu et al.
# They first extracted keywords from Influential Users (IUs). Assign tf-idf weights to the words of tweets.
# Extract frequent word sets. Score every tweets based on tf-idf. Rank tweets based on their scores.

# Collect all tweets from IUs that most frequent keywords were selected from.
client = MongoClient()
db = client['twitter']
Dic_Coll = db['Dic_Coll']  # Collection with all IU users' tweets

collection = list(db.Dic_Coll.find())
IUs_tweet = []
for tweet in collection:
    IUs_tweet.append(tweet['text'].lower().split())

# The paper has selected 3200 tweets for IUs.
IUs_tweet = IUs_tweet[:3200]
# Extracted keywords from IUs.
# This can be done in two ways: Extracted keywords from my IUs list. Or simply keywords reported in paper.
with open('All-TI-Keywords-Weights.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    keywords_sina = []
    for tweet in reader:
        keywords_sina.append(tweet[0])
keywords_sina = keywords_sina[:50]

# Keywords from paper
paper_keywords = ['dctraffic', 'mdtraffic', 'vatraffic', 'crash', 'due', 'delays', 'st', 'traffic', 'right', 'ave',
                  'lane', 'lanes', 'rd', 'nw', 'accident', 'buses', 'left', 'md', 'nb', 'bridge', 'sb', 'earlier',
                  'street', 'near' 'blocked','loop', 'congestion', 'expect', 'va', 'dc', 'update', 'road', 'work',
                  'following', 'closed', 'open', 'vehicle', 'inner', 'car', 'killed', 'new', 'get', 'minute',
                  'directions', 'close', 'schedule', 'police', 'beltway', 'operating', 'us']

'''
# Applying the method proposed in the study
def make_query(high_words, num_comb, low_itemset):
    """
    :param keywords_highSupport: The words (one word) with high support in the IUs tweet
    :param num_comb: number of words in words combination (>=2)
    :return: make a combination of words based on number of keywords and the previous low itemsets
    """
    query = list(combinations(high_words, num_comb))
    for comb in query:
        for lowcomb in low_itemset:
            if all(word in comb for word in lowcomb):
                query.remove(comb)
                break
    return query

def low_high_comb(query, IUs_tweet, min_support=0.02):
    """
    :param IUs_tweet:
    :param min_support: 0.02
    :return: itemset that their support are either greater or less than min_support
    """
    count_keywords = []

    for comb in query:
        counter = 0
        for tweet in IUs_tweet:
            if all(word in ' '.join(tweet) for word in comb):
                counter += 1
        count_keywords.append(counter)
    support = [item * 1./len(IUs_tweet) for item in count_keywords]
    high_itemset = [query[i] for i, item in enumerate(support) if item > min_support]
    low_itemset = [query[i] for i, item in enumerate(support) if item <= min_support]
    return high_itemset, low_itemset

# Apply Apriori algorithm, find words with high support
count_keywords = []
for word in keywords_sina:
    counter = 0
    for tweet in IUs_tweet:
        if word in tweet:
            counter += 1
    count_keywords.append(counter)
support = [item * 1./len(IUs_tweet) for item in count_keywords]
min_support = 0.02
# high-support item sets with only one word.
high_words = [keywords_sina[i] for i, item in enumerate(support) if item > min_support]
low_itemset = [keywords_sina[i] for i, item in enumerate(support) if item <= min_support]

all_high_itemset = []
num_comb = 2
high_itemset = high_words.copy()
# We only find wordset with three word combination as said in paper.
for _ in range(3):
    all_high_itemset.append(high_itemset)
    query = make_query(high_words, num_comb, low_itemset)
    high_itemset, low_itemset = low_high_comb(query, IUs_tweet, min_support=0.02)
    num_comb += 1


with open('keywords_Fu_Study.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(all_high_itemset, f)
'''
with open('keywords_Fu_Study.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    all_high_itemset = pickle.load(f)
# Extracting tweets from test set based on frequent three-word combination
filename = '../LSTM-CNN code/1_TestSet_2Class.csv'
with open(filename, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    test_tweets = []
    for tweet in reader:
        test_tweets.append(tweet)

itemset = all_high_itemset[2]
extracted_tweet = []
non_extracted_tweet = []
for tweet in test_tweets:
    extract = 0
    for word_comb in itemset:
        if all(word in tweet[2].lower() for word in word_comb):
            extracted_tweet.append(tweet)
            extract = 1
            break
    if extract == 0:
        non_extracted_tweet.append(tweet)

vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word', vocabulary=keywords_sina)
# Convert the tokenized vectors to bag of word representation with only term frequency.
tweet_corpus = [tweet[2] for tweet in extracted_tweet]
features_counts = vectorizer.fit_transform(tweet_corpus)
# Convert bag of words with term frequency to tf-idf as new weights.
transformer = TfidfTransformer(smooth_idf=False)
features_tfidf = transformer.fit_transform(features_counts)
sum_tfidf = np.sum(features_tfidf, axis=1)

# The percentile value is the threshold for selecting traffic tweets from those extracted tweets.
# based on the paper it is better to put it az zero. I.e., all extracted tweets are traffic-related
percentile = np.percentile(sum_tfidf, 0)
predicted_traffic_tweet = [extracted_tweet[i] for i, item in enumerate(sum_tfidf) if item > percentile]
predicted_non_traffic_tweet = [extracted_tweet[i] for i, item in enumerate(sum_tfidf) if item <= percentile]

counter_traffic = sum(1 for tweet in predicted_traffic_tweet if int(tweet[0]) == 1)
counter_non_traffic = sum(1 for tweet in predicted_non_traffic_tweet if int(tweet[0]) == 0)

counter_non_traffic += sum(1 for tweet in non_extracted_tweet if int(tweet[0]) == 0)

accuracy = (counter_non_traffic + counter_traffic) * 1. / len(test_tweets)
print('Fu accuracy', accuracy)
