from pymongo import MongoClient
import tweepy
from langid.langid import LanguageIdentifier, model
import re
import csv
import numpy as np
import math

# This file only collects unlabeled data for labelling. Data are absolutely collected by random with no filtering

# Connecting to MongoDB, creating twitter database, and tweets collection
client = MongoClient()
db = client['twitter']  # or access by attribute style rather than dictionary style: db = client.twitter
Initial_Coll = db['Initial_Coll']
Final_Coll = db['Final_Coll']
temporary = db['temporary']
Manual_Labeled_15000 = db['Manual_Labeled_15000']

# Authentication process to Twitter API
auth = tweepy.OAuthHandler('F9av1j73aaAKKmLwSOw0wUUD8', 'ulslZZoPRVT4atRWvKIX4VQmFnQv3fodBdWT5q6Q2KKp1h5a2Q')
auth.set_access_token('3405607174-us6ym0neIg8vyglbVzISa7weoe68qEuqpsBcupi',
                      'nDUawu1zuv3VzUt1FGis4zc1nTN1vsiFLiaeMGpPLgZMw')
api = tweepy.API(auth, compression=False, wait_on_rate_limit=False)

# Update the tweets' labels in the main collection: Manual_Labeled_15000
'''''
with open('15000-Tweets-For-Labeling-new.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        #-1 is assigned to those tweets that creates ambiguity. So we remove them at the end
        result = db.Manual_Labeled_15000.update_one({'id_str': row[1]}, {'$set': {'label': row[0]}})
'''''

ExceedRateLimit = []
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
db.Initial_Coll.delete_many({})

try:
    for t in tweepy.Cursor(api.search, geocode='39.833333,-98.583333,1700mi', lang='en').items(150):
        if identifier.classify(t.text)[0] != 'en':
            continue
        else:
            ID = 's' + t.id_str
            attributes = {'User_screen_name': t.user.screen_name, 'combo_no': 'Random', 'word_combo': 'Random',
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

        db.Initial_Coll.insert_one(attributes)

except tweepy.error.TweepError:
    ExceedRateLimit.append(id)

### On a new collection (Dic_Coll): records a tweet only once. I.e. there is no re-tweets
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
with open('1-15000-Tweets-For-Labeling-random.csv', 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in Coll:
        writer.writerow(['', tweet['id_str'], tweet['text']])


