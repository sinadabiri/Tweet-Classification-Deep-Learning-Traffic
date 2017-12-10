from pymongo import MongoClient
import tweepy
from langid.langid import LanguageIdentifier, model
import re
import csv
import numpy as np
import math
import os

os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\TI+NTI Data Collection+Classification')

# Connecting to MongoDB, creating twitter database, and tweets collection
client = MongoClient()
db = client['twitter']
# Manual_Labeled_15000: the collection from the first dataset contains Traffic related and non-traffic related tweets
Manual_Labeled_15000 = db['Manual_Labeled_15000']
# TI_IUs_Tweets: the collection from the second dataset with more traffic recurring tweets collected from IUs
TI_IUs_Tweets = db['TI_IUs_Tweets']
# NTI_Tweets: the collection from the third dataset contains only non-related traffic tweets collected from non-IUs
NTI_Tweets = db['NTI_Tweets']

# all_tweets: the collection that contains all tweets from the three data sets.
all_tweets = db['all_tweets']
''''
Collection_1 = db.Manual_Labeled_15000.find()
db.all_tweets.insert_many(Collection_1)
Collection_2 = db.TI_IUs_Tweets.find()
db.all_tweets.insert_many(Collection_2)
Collection_3 = db.NTI_Tweets.find()
db.all_tweets.insert_many(Collection_3)
'''
collection = db.all_tweets.find()
with open('Labeled_All Tweets_3Class.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in collection:
        writer.writerow([int(tweet['label']), tweet['text']])
collection = db.all_tweets.find()
with open('Labeled_All Tweets_2Class.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in collection:
        if int(tweet['label']) == 1:
            writer.writerow([1, tweet['text']])
        else:
            writer.writerow([2, tweet['text']])

