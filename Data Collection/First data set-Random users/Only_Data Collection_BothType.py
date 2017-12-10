from pymongo import MongoClient
import tweepy
from langid.langid import LanguageIdentifier, model
import re
import csv
import numpy as np
import math

# This file only collects unlabeled data for labelling. Data are collected based on diff combination of TI keywrods.

# Connecting to MongoDB, creating twitter database, and tweets collection
client = MongoClient()
db = client['twitter']  # or access by attribute style rather than dictionary style: db = client.twitter
Initial_Coll = db['Initial_Coll']
temporary = db['temporary']
Manual_Labeled_15000 = db['Manual_Labeled_15000']

# Authentication process to Twitter API
auth = tweepy.OAuthHandler('F9av1j73aaAKKmLwSOw0wUUD8', 'ulslZZoPRVT4atRWvKIX4VQmFnQv3fodBdWT5q6Q2KKp1h5a2Q')
auth.set_access_token('3405607174-us6ym0neIg8vyglbVzISa7weoe68qEuqpsBcupi',
                      'nDUawu1zuv3VzUt1FGis4zc1nTN1vsiFLiaeMGpPLgZMw')
api = tweepy.API(auth, compression=False, wait_on_rate_limit=False)


'''''
# Update the tweets' labels in the main collection: Manual_Labeled_15000
# Note: 15000_Final_Mixed_Labeled.csv: this file is subjected to filter function after getting mixed with new and random
with open('2-15000_Final_Mixed_Labeled.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        #-1 is assigned to those tweets that creates ambiguity. So we remove them at the end
        result = db.Manual_Labeled_15000.update_one({'id_str': row[1]}, {'$set': {'label': row[0]}})
'''

# This is the loop over specified query, search API
with open('All-TI-Keywords-Weights.csv', 'r', newline='') as f:
    reader = csv.reader(f)
    keywords = []
    for row in reader:
        keywords.append(row[0])

# Combine the keywords with some more useful TI words, which are obtained based on our observation.
keywords = keywords[:50] + ['delays', 'heavy', 'slow', 'jammed', 'Traffic Advisory']
ExceedRateLimit = []
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
No_Tweets_eachCombo = []
# word_Combo=first number: no. of word combination
word_Combo = [2, 3, 4]
no_tweet = 150
portion = [1./2, 1./3, 1./6]
Number = np.random.randint(low=0, high=2, size=(2, 10))
db.Initial_Coll.delete_many({})
WordComboCounter = []
for j in range(len(word_Combo)):
    Number = np.random.randint(low=0, high=len(keywords), size=(word_Combo[j], math.ceil(portion[j] * no_tweet)))
    for z in range(len(Number[0, :])):
        query = []
        for k in range(word_Combo[j]):
            query.append(keywords[Number[k, z]])
        query = " ".join(query)
        try:
            count = 0
            # geocode: specify the US area. It overlaps a bit with north of Canada
            for t in tweepy.Cursor(api.search, q=query, geocode='39.833333,-98.583333,1700mi', lang='en').items(1):
                if identifier.classify(t.text)[0] != 'en':
                    continue
                else:
                    count += 1
                    ID = 's' + t.id_str
                    attributes = {'User_screen_name': t.user.screen_name, 'combo_no': j, 'word_combo': query,
                                  'created_at': t.created_at, 'id_str': ID, 'label': '-1'}
                    if 'retweeted_status' in dir(t):
                        attributes.update({'original_text': t.retweeted_status.text})
                        text = re.sub(r'http\S+', '', t.retweeted_status.text)
                        text = re.sub(r'@\S+', '', text)
                        text = re.sub(r'\b\d+\b', 'number', text)
                        text = re.sub(r'I-number', 'highway', text)
                        text = re.sub(r'i-number', 'highway', text)
                        text = re.sub(r'US-number', 'highway', text)
                        text = text.replace('US number', 'highway')
                        text = text.replace('U.S. number', 'highway')
                        text = text.replace('u.s. number', 'highway')
                        text = text.replace('us number', 'highway')
                        text = text.replace('number', '')
                        text = text.replace('  ', ' ')
                        attributes.update({'text': text})
                    else:
                        attributes.update({'original_text': t.text})
                        text = re.sub(r'http\S+', '', t.text)
                        text = re.sub(r'@\S+', '', text)
                        text = re.sub(r'\b\d+\b', 'number', text)
                        text = re.sub(r'I-number', 'highway', text)
                        text = re.sub(r'i-number', 'highway', text)
                        text = re.sub(r'US-number', 'highway', text)
                        text = text.replace('US number', 'highway')
                        text = text.replace('U.S. number', 'highway')
                        text = text.replace('u.s. number', 'highway')
                        text = text.replace('us number', 'highway')
                        text = text.replace('number', '')
                        text = text.replace('  ', ' ')
                        attributes.update({'text': text})

                    if t.place is None or t.place.bounding_box is None:
                        attributes.update({'place': None})
                    else:
                        attributes.update({'place': t.place.bounding_box.coordinates})
                    if t.coordinates is None:
                        attributes.update({'coordinates': None})
                    else:
                        attributes.update({'coordinates': t.coordinates['coordinates']})

                WordComboCounter.append(count)
                db.Initial_Coll.insert_one(attributes)

        except tweepy.error.TweepError:
            ExceedRateLimit.append(id)

# Records a tweet only once. I.e. there is no re-tweets
Collection = db.Initial_Coll.find()
db.temporary.delete_many({})
for tweet in Collection:
    if db.Initial_Coll.count() == 0:
        break
    if db.Manual_Labeled_15000.find({'text': tweet['text']}).count() == 0:
        db.temporary.insert_one(tweet)
        db.Manual_Labeled_15000.insert_one(tweet)
        db.Initial_Coll.delete_many({'text': tweet['text']})

Coll = db.temporary.find({})
with open('1-15000-Tweets-For-Labeling-new.csv', 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in Coll:
        writer.writerow(['', tweet['id_str'], tweet['text']])


