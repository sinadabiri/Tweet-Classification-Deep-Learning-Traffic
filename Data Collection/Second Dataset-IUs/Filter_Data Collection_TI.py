import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import csv
import os
from pymongo import MongoClient
from gensim.models.keyedvectors import KeyedVectors
import random

# This file filters out tweets with the same words regardless of the word ordering in a tweet.
# Words in a tweet are determined based on Google pre-trained word2vec.

os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\TI-Data Collection')
client = MongoClient()
db = client['twitter']  # or access by attribute style rather than dictionary style: db = client.twitter
Initial_Coll_TI = db['Initial_Coll_TI']
TI_IUs_Tweets = db['TI_IUs_Tweets']
temporary_TI = db['temporary_TI']

'''''
# Remove tweets with less than four tokens, excluding the stop words. 
with open('Final_Labeled_DataCollection-TI_Tweets.csv', 'r', newline='', encoding='utf-8') as f:
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

with open('Final_Labeled_DataCollection-TI_Tweets.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for tweet in Tweets:
        if len(analyze(tweet[2])) >= 4 and tweet[0] != '-1':
            writer.writerow(tweet)
'''

with open('Final_Labeled_DataCollection-TI_Tweets.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    TI_Tweets = []
    TI_Whole_Tweets = []
    for row in reader:
        TI_Whole_Tweets.append(row)
        TI_Tweets.append(row[2])

Google_Word2Vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, encoding='latin-1')

# We tokenize the text into words. Stop words and word not including in Google are discarded.
vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
analyze = vectorizer.build_analyzer()
miss_words = []
Total_Tweet_Vec_List = []
for t in TI_Tweets:
    Name = analyze(t)
    Tweet_Vec_List = []
    for word in Name:
        try:
            wordVec = Google_Word2Vec[word]
            Tweet_Vec_List.append(word)
        except KeyError:
            miss_words.append(word)
    Total_Tweet_Vec_List.append(Tweet_Vec_List)

New_TI_Tweets = []
for tweet in Total_Tweet_Vec_List:
    New_TI_Tweets.append(' '.join(tweet))

DocumentTerm1 = vectorizer.fit_transform(New_TI_Tweets)
DocumentTerm_Array1 = DocumentTerm1.toarray()

u, indices = np.unique(DocumentTerm_Array1, return_index=True, axis=0)

# Here, we write the unique tweets in a csv file, which may be discarded according to the need.
with open('Final_Labeled_DataCollection-TI_Tweets.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for i in indices:
        writer.writerow(TI_Whole_Tweets[i])
