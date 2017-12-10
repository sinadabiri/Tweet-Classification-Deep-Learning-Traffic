import tweepy
from pymongo import MongoClient
import os
import re
from langid.langid import LanguageIdentifier, model
import csv
import random
os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\TI-Data Collection')


# Authentication process to Twitter API
auth = tweepy.OAuthHandler('F9av1j73aaAKKmLwSOw0wUUD8', 'ulslZZoPRVT4atRWvKIX4VQmFnQv3fodBdWT5q6Q2KKp1h5a2Q')
auth.set_access_token('3405607174-us6ym0neIg8vyglbVzISa7weoe68qEuqpsBcupi',
                      'nDUawu1zuv3VzUt1FGis4zc1nTN1vsiFLiaeMGpPLgZMw')
api = tweepy.API(auth, compression=False, wait_on_rate_limit=False)

# Connecting to MongoDB, creating twitter database, and tweets collection
client = MongoClient()
db = client['twitter']  # or access by attribute style rather than dictionary style: db = client.twitter
Initial_Coll_TI = db['Initial_Coll_TI']
TI_IUs_Tweets = db['TI_IUs_Tweets']
temporary_TI = db['temporary_TI']

# Remove Extra tweets from the main Coll. I.e, tweets that dont exist in the CSV file
''''
with open('Final_wo-1_DataCollection-TI_Tweets.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    ID_CSV = []
    for row in reader:
        ID_CSV.append(row[1])

Coll = db.TI_IUs_Tweets.find({})
ID_Coll = []
for i in Coll:
    ID_Coll.append(i['id_str'])

Remove_ID = list(set(ID_Coll) - set(ID_CSV))
for ID in Remove_ID:
    db.TI_IUs_Tweets.delete_one({'id_str': ID})

'''
# Update the tweets' labels in the main collection: TI_IUs_Tweets
'''''
with open('1_Final_DataCollection-TI_IUs_Tweets.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        #-1 is assigned to those tweets that creates ambiguity. So we remove them at the end
        result = db.TI_IUs_Tweets.update_one({'id_str': row[1]}, {'$set': {'label': row[0]}})
db.TI_IUs_Tweets.delete_many({'id_str': '-1'})
db.TI_IUs_Tweets.delete_many({'id_str': -1})
'''''
'''''
with open('1_Final_DataCollection-TI_IUs_Tweets.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    TI_Tweets = []
    for row in reader:
        TI_Tweets.append(row)
with open('Final_wo-1_DataCollection-TI_Tweets.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in TI_Tweets:
        if tweet[0] != '-1':
            writer.writerow(tweet)
'''
ExceedRateLimit = []
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

# Retrieve the TI_IUs accounts
TI_IUs_username = []
with open('IU_511+DOT+Search_Tweepy.txt', 'r', newline='') as f:
    reader = csv.reader(f)
    for account in reader:
        TI_IUs_username.append(account[0])

for i in TI_IUs_username[0:10]:
    Initial_List_TI = []
    try:
        for page in tweepy.Cursor(api.user_timeline, screen_name=i).pages(10):
            for t in page:
                if identifier.classify(t.text)[0] != 'en':
                    continue
                else:
                    ID = 's' + t.id_str
                    attributes = {'User_screen_name': t.user.screen_name,
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

                Initial_List_TI.append(attributes)

        random.shuffle(Initial_List_TI)
        if len(Initial_List_TI) > 100:
            db.Initial_Coll_TI.insert_many(Initial_List_TI[:100])
        else:
            db.Initial_Coll_TI.insert_many(Initial_List_TI)

    except tweepy.error.TweepError:
        ExceedRateLimit.append(id)

# records a tweet only once. I.e. there is no re-tweets
Collection = db.Initial_Coll_TI.find()
db.temporary_TI.delete_many({})
for tweet in Collection:
    if db.Initial_Coll_TI.count() == 0:
        break
    if db.TI_IUs_Tweets.find({'text': tweet['text']}).count() == 0:
        db.temporary_TI.insert_one(tweet)
        db.TI_IUs_Tweets.insert_one(tweet)
        db.Initial_Coll_TI.delete_many({'text': tweet['text']})

Coll = db.temporary_TI.find({})
with open('DataCollection-TI_IUs_Tweets.csv', 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in Coll:
        writer.writerow(['', tweet['id_str'], tweet['text']])
