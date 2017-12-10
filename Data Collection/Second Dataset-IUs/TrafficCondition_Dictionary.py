import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import csv
import os
from pymongo import MongoClient
from gensim.models.keyedvectors import KeyedVectors


os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\TI-Data Collection')
client = MongoClient()
db = client['twitter']  # or access by attribute style rather than dictionary style: db = client.twitter
Initial_Coll_TI = db['Initial_Coll_TI']
TI_IUs_Tweets = db['TI_IUs_Tweets']
temporary_TI = db['temporary_TI']


# Collect all TI tweets from Final_Coll collection and insert them into tweet_corpus list for vectorization process
with open('15000-Tweets-For-Labeling-new.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    tweet_corpus = []
    for row in reader:
        if row[0] == '3':
            tweet_corpus.append(row[2])

# Vectorization, finding normalized TF for each word, and building the TI dictionary
vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
DocumentTerm = vectorizer.fit_transform(tweet_corpus)
DocumentTerm_Array = DocumentTerm.toarray()
Name = vectorizer.get_feature_names()
Frequency = np.sum(DocumentTerm_Array, axis=0)
Sum = np.sum(Frequency)
Weights = []
[Weights.append((i - min(Frequency))/(max(Frequency) - min(Frequency))) for i in Frequency]

Sort_Index = np.argsort(Frequency)
Sort_Index = Sort_Index[::-1]
Sort_Weights = []
[Sort_Weights.append(Weights[i]) for i in Sort_Index]

Sorted_All_Name_Weights = []
X = []
NOofwords = 50
for i in range(NOofwords):
    Sorted_All_Name_Weights.append(Name[Sort_Index[i]])

with open('All-TI-Keywords-Weights.csv', 'r', newline='') as f:
    reader = csv.reader(f)
    TI_keywords = []
    for row in reader:
        TI_keywords.append(row[0])

with open('TrafficConditions_Keywords_Weights.csv', 'w', newline='') as handle:
    writer = csv.writer(handle)
    for row in Sorted_All_Name_Weights:
        writer.writerow([row])
