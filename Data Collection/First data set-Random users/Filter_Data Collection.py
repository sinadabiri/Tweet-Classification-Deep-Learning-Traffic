import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import csv
import os
from pymongo import MongoClient
from gensim.models.keyedvectors import KeyedVectors
import random

# This file filters out tweets with the same words regardless of the word ordering in a tweet.
# Words in a tweet are determined based on Google pre-trained word2vec.

os.chdir(r'C:\Users\Sina\Desktop\Heaslipaa\Traffic Incident Detection-Twitter\Codes')

with open('15000_Final_Mixed_Labeled.csv', 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    Tweets = []
    Whole_Tweets = []
    for row in reader:
        Whole_Tweets.append(row)
        Tweets.append(row[2])

Google_Word2Vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, encoding='latin-1')

vectorizer = CountVectorizer(min_df=1, stop_words='english', ngram_range=(1, 1), analyzer=u'word')
analyze = vectorizer.build_analyzer()
miss_words = []
Total_Tweet_Vec_List = []
for t in Tweets:
    Name = analyze(t)
    Tweet_Vec_List = []
    for word in Name:
        try:
            wordVec = Google_Word2Vec[word]
            Tweet_Vec_List.append(word)
        except KeyError:
            miss_words.append(word)
    Total_Tweet_Vec_List.append(Tweet_Vec_List)

New_Tweets = []
for tweet in Total_Tweet_Vec_List:
    New_Tweets.append(' '.join(tweet))

DocumentTerm1 = vectorizer.fit_transform(New_Tweets)
DocumentTerm_Array1 = DocumentTerm1.toarray()

u, indices = np.unique(DocumentTerm_Array1, return_index=True, axis=0)

# Here, we write the unique tweets in a csv file, which may be discarded according to the need.
with open('15000_Final_Mixed_Labeled.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for i in indices:
        writer.writerow(Whole_Tweets[i])

