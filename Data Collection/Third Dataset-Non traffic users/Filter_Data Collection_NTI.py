import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import csv
import os
from pymongo import MongoClient
from gensim.models.keyedvectors import KeyedVectors
import random

# This file filters out tweets with the same words regardless of the word ordering in a tweet.
# Words in a tweet are determined based on Google pre-trained word2vec.

os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\NTI Data Collection')

# Connecting to MongoDB, creating twitter database, and tweets collection
client = MongoClient()
db = client['twitter']  # or access by attribute style rather than dictionary style: db = client.twitter
Initial_Coll_NTI = db['Initial_Coll_NTI']
NTI_Tweets = db['NTI_Tweets']
temporary_NTI = db['temporary_NTI']

'''''
# Remove tweets with less than four tokens, excluding the stop words. 
with open('Final_Labeled_DataCollection-NTI_Tweets.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    Tweets = []
    for row in reader:
        Tweets.append(row)

vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
analyze = vectorizer.build_analyzer()
count = 0
for i in range(len(Tweets)):
    if len(analyze(Tweets[i][2])) < 4:
        count += 1

with open('Final_Labeled_DataCollection-NTI_Tweets.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in Tweets:
        if len(analyze(tweet[2])) >= 4 and tweet[0] != '-1':
            writer.writerow(tweet)
'''''

with open('Final_Labeled_DataCollection-NTI_Tweets.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    NTI_Tweets_1 = []
    NTI_Whole_Tweets = []
    for row in reader:
        NTI_Whole_Tweets.append(row)
        NTI_Tweets_1.append(row[2])

Google_Word2Vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, encoding='latin-1')

vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
analyze = vectorizer.build_analyzer()
miss_words = []
Total_Tweet_Vec_List = []
for t in NTI_Tweets_1:
    Name = analyze(t)
    Tweet_Vec_List = []
    for word in Name:
        try:
            wordVec = Google_Word2Vec[word]
            Tweet_Vec_List.append(word)
        except KeyError:
            miss_words.append(word)
    Total_Tweet_Vec_List.append(Tweet_Vec_List)

New_NTI_Tweets = []
for tweet in Total_Tweet_Vec_List:
    New_NTI_Tweets.append(' '.join(tweet))

DocumentTerm1 = vectorizer.fit_transform(New_NTI_Tweets)
DocumentTerm_Array1 = DocumentTerm1.toarray()

u, indices = np.unique(DocumentTerm_Array1, return_index=True, axis=0)

# Here, we write the unique tweets in a csv file, which may be discarded according to the need.
with open('Final_Labeled_DataCollection-NTI_Tweets.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for i in indices:
        writer.writerow(NTI_Whole_Tweets[i])

