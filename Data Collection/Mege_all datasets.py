from pymongo import MongoClient
import csv
import random

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
'''
# Update the labels of wrong classified tweets.
with open('1_wrong_classified_tweets.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        db.all_tweets.update_one({'id_str': row[3]}, {'$set': {'label': row[2]}})
        db.Manual_Labeled_15000.update_one({'id_str': row[3]}, {'$set': {'label': row[2]}})
        db.TI_IUs_Tweets.update_one({'id_str': row[3]}, {'$set': {'label': row[2]}})
        db.NTI_Tweets.update_one({'id_str': row[3]}, {'$set': {'label': row[2]}})
'''

Collection_1 = db.Manual_Labeled_15000.find({'label': '3'}).count()
#db.all_tweets.insert_many(Collection_1)
Collection_2 = db.TI_IUs_Tweets.find({'label': '3'}).count()
#db.all_tweets.insert_many(Collection_2)
Collection_3 = db.NTI_Tweets.find({'label': '3'}).count()
#db.all_tweets.insert_many(Collection_3)
a = 2

A = db.all_tweets.find({'label': '1'}).count()
A1 = db.all_tweets.find({'label': '2'}).count()
A2 = db.all_tweets.find({'label': '3'}).count()


# create 2-class and 3-class csv files for my work
# Use all_tweets for creating data from all labeled tweets or any other collection.
collection = db.Manual_Labeled_15000.find()
tweet_0 = []
tweet_1 = []
tweet_2 = []
tweet_3class_count = []
for tweet in collection:
    if int(tweet['label']) == 1:
        tweet_0.append([0, tweet['id_str'], tweet['text']])
    if int(tweet['label']) == 2:
        tweet_1.append([1, tweet['id_str'], tweet['text']])
    if int(tweet['label']) == 3:
        tweet_2.append([2, tweet['id_str'], tweet['text']])
# the following for creating tweet from all collections with adjusting them to have the same portion of
# traffic and non-traffic tweets.
tweet_0 = tweet_0[:25550]
tweet_1 = tweet_1[:17437]
random.seed(7)
random.shuffle(tweet_0)
random.shuffle(tweet_1)
random.shuffle(tweet_2)

training_tweets = tweet_0[:int(.8 * len(tweet_0))] + tweet_1[:int(.8 * len(tweet_1))] + tweet_2[:int(.8 * len(tweet_2))]
test_tweets = tweet_0[int(.8 * len(tweet_0)):] + tweet_1[int(.8 * len(tweet_1)):] + tweet_2[int(.8 * len(tweet_2)):]
random.shuffle(training_tweets)
random.shuffle(test_tweets)

with open('OnlyFirstSet_TrainingSet_3Class.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in training_tweets:
        writer.writerow([tweet[0], tweet[1], tweet[2]])

with open('OnlyFirstSet_TestSet_3Class.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in test_tweets:
        writer.writerow([tweet[0], tweet[1], tweet[2]])

with open('OnlyFirstSet_TrainingSet_2Class.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in training_tweets:
        if tweet[0] == 0:
            writer.writerow([0, tweet[1], tweet[2]])
        else:
            writer.writerow([1, tweet[1], tweet[2]])

with open('OnlyFirstSet_TestSet_2Class.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in test_tweets:
        if tweet[0] == 0:
            writer.writerow([0, tweet[1], tweet[2]])
        else:
            writer.writerow([1, tweet[1], tweet[2]])


