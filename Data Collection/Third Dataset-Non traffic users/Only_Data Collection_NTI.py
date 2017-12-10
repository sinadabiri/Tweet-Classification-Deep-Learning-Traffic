import tweepy
from pymongo import MongoClient
import os
import re
from langid.langid import LanguageIdentifier, model
import itertools
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import random

os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\NTI Data Collection')

# Authentication process to Twitter API
auth = tweepy.OAuthHandler('F9av1j73aaAKKmLwSOw0wUUD8', 'ulslZZoPRVT4atRWvKIX4VQmFnQv3fodBdWT5q6Q2KKp1h5a2Q')
auth.set_access_token('3405607174-us6ym0neIg8vyglbVzISa7weoe68qEuqpsBcupi',
                      'nDUawu1zuv3VzUt1FGis4zc1nTN1vsiFLiaeMGpPLgZMw')
api = tweepy.API(auth, compression=False, wait_on_rate_limit=False)

# Connecting to MongoDB, creating twitter database, and tweets collection
client = MongoClient()
db = client['twitter']  # or access by attribute style rather than dictionary style: db = client.twitter
Initial_Coll_NTI = db['Initial_Coll_NTI']
NTI_Tweets = db['NTI_Tweets']
temporary_NTI = db['temporary_NTI']


# Remove Extra tweets from the main Coll. I.e, tweets that dont exist in the CSV file
'''''
with open('Final_Labeled_DataCollection-NTI_Tweets.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    ID_CSV = []
    for row in reader:
        ID_CSV.append(row[1])

Coll = db.NTI_Tweets.find({})
ID_Coll = []
for i in Coll:
    ID_Coll.append(i['id_str'])

Remove_ID = list(set(ID_Coll) - set(ID_CSV))
for ID in Remove_ID:
    db.NTI_Tweets.delete_one({'id_str': ID})

'''''
# Update the tweets' labels in the main collection: NTI_Tweets
'''''
with open('Labeled_DataCollection-NTI_Tweets.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        #-1 is assigned to those tweets that creates ambiguity. So we remove them at the end
        result = db.NTI_Tweets.update_one({'id_str': row[1]}, {'$set': {'label': row[0]}})
'''

vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
analyze = vectorizer.build_analyzer()

ExceedRateLimit = []
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

# Collecting NTI accounts
NTI_Account = ['AshleyFurniture', 'uhaul', 'AEO', 'TommyHilfiger', 'nabp', 'MikePenceVP', 'JLo', 'Cartier',
               'verizon', 'sheratonhotels']
NTI_Account_1 = ['USDA', 'Foodservicecom', 'CommerceGov', 'USAlcoholPolicy', 'FHFA', 'USClubSoccer'
                 , 'ThePSBC', 'AppalachianPowe', 'katyperry', 'taylorswift13']
NTI_Account_2 = ['usafootball', 'justinbieber', 'ATTCares', 'facebook', 'realDonaldTrump', 'JohnKerry',
                 'ChemistryALevel', 'RalphLauren', 'JournoResource', 'OldNavy']

NTI_Account_3 = ['virginia_tech', 'MercedesBenz', 'BarackObama', 'jimmyfallon', 'KingJames', 'Oprah', 'chevrolet'
                 'WellsFargo', 'Aetna', 'GEICO']
NTI_Account_4 = ['Dell', 'SamsungMobileUS', 'ElsevierConnect', 'UNICEF', 'OneWTC', 'McDonalds', 'StarwoodBuzz',
                 'BASF', 'KelloggCompany', 'GeneralMills']
NTI_Account_5 = ['BillGates', 'LockheedMartin', 'GenslerOnWork', 'SamsoniteUSA', '_AnimalAdvocate', 'UN', 'usabasketball'
                 'HospInnovations', 'costco', 'ChooseChicago']
# I iterate over NTI_Account_i each time to collect data from a vareity of NTI accounts.

for i in NTI_Account:
    Initial_List_NTI = []
    try:
        for page in tweepy.Cursor(api.user_timeline, screen_name=i).pages(30):
            for t in page:
                if identifier.classify(t.text)[0] != 'en':
                    continue
                else:
                    ID = 's' + t.id_str
                    attributes = {'User_screen_name': t.user.screen_name,
                                  'created_at': t.created_at, 'id_str': ID, 'label': -1}
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

                Initial_List_NTI.append(attributes)

        random.shuffle(Initial_List_NTI)
        if len(Initial_List_NTI) > 300:
            db.Initial_Coll_NTI.insert_many(Initial_List_NTI[:300])
        else:
            db.Initial_Coll_NTI.insert_many(Initial_List_NTI)

    except tweepy.error.TweepError:
        ExceedRateLimit.append(id)

# records a tweet only once. I.e. there is no re-tweets
Collection = db.Initial_Coll_NTI.find()
db.temporary_NTI.delete_many({})
for tweet in Collection:
    if db.Initial_Coll_NTI.count() == 0:
        break
    if db.NTI_Tweets.find({'text': tweet['text']}).count() == 0:
        db.temporary_NTI.insert_one(tweet)
        db.NTI_Tweets.insert_one(tweet)
        db.Initial_Coll_NTI.delete_many({'text': tweet['text']})

Coll = db.temporary_NTI.find({})
with open('1-DataCollection-NTI_Tweets.csv', 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in Coll:
        writer.writerow(['', tweet['id_str'], tweet['text']])
